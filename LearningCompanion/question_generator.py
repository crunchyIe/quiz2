import re
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    # Force download all required resources
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Specifically needed for tokenizers/punkt_tab/english/
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')  # Also needed for English POS tagging
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {str(e)}")

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_important_sentences(text, top_n=50):
    """Extract the most important sentences from text using TF-IDF
    
    This function prioritizes sentences that contain key information based on
    their TF-IDF scores, making it optimal for question generation.
    
    Args:
        text: The document text to analyze
        top_n: The number of important sentences to extract
        
    Returns:
        List of important sentences from the text
    """
    # Preprocess and split into sentences
    sentences = sent_tokenize(text)
    
    # Log sentence count
    logger.info(f"Document contains {len(sentences)} total sentences")
    
    # For very large documents, we'll need to chunk the processing
    if len(sentences) > 1000:
        logger.info("Large document detected, processing in chunks")
        # Process document in chunks of 1000 sentences
        chunks = [sentences[i:i+1000] for i in range(0, len(sentences), 1000)]
        important_from_chunks = []
        
        # Process each chunk to get important sentences
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_preprocessed = [preprocess_text(sentence) for sentence in chunk]
            
            # Filter out very short sentences (likely not informative)
            valid_indices = [i for i, s in enumerate(chunk_preprocessed) if len(s.split()) > 5]
            valid_chunk = [chunk[i] for i in valid_indices]
            
            # Take proportional number of sentences from each chunk
            chunk_top_n = max(5, int(top_n * len(chunk) / len(sentences)))
            important_from_chunks.extend(
                _process_sentence_chunk(valid_chunk, chunk_top_n)
            )
        
        # Take the top_n sentences from all chunks
        if len(important_from_chunks) > top_n:
            # Shuffle to ensure variety, but limit to top_n
            random.shuffle(important_from_chunks)
            return important_from_chunks[:top_n]
        return important_from_chunks
    
    # For smaller documents, process all at once
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Extract key terminology from the entire document
    # This will help us identify sentences with domain-specific terminology
    document_text = ' '.join(preprocessed_sentences)
    document_words = word_tokenize(document_text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Count word frequencies
    word_freq = {}
    for word in document_words:
        if len(word) > 3 and word not in stop_words and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Extract top keywords based on frequency
    keyword_threshold = max(5, int(len(word_freq) * 0.02))  # Top 2% of words
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:keyword_threshold]
    important_keywords = {word for word, freq in keywords}
    
    # Prioritize sentences with important domain terminology
    # Score each sentence based on keyword coverage
    keyword_scores = []
    for i, sentence in enumerate(preprocessed_sentences):
        if len(sentence.split()) <= 5:  # Skip very short sentences
            keyword_scores.append((i, 0))
            continue
            
        sentence_words = set([w.lower() for w in word_tokenize(sentence) if w.isalpha()])
        # Score is the count of important keywords in this sentence
        keyword_count = len(sentence_words.intersection(important_keywords))
        keyword_scores.append((i, keyword_count))
    
    # Sort by keyword coverage and get indices of top sentences
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    keyword_rich_indices = [idx for idx, score in keyword_scores[:int(top_n * 1.5)] if score > 0]
    
    # Filter out very short sentences (likely not informative)
    valid_indices = [i for i, s in enumerate(preprocessed_sentences) if len(s.split()) > 5]
    valid_sentences = [sentences[i] for i in valid_indices]
    
    if len(valid_sentences) <= top_n:
        return valid_sentences
    
    # Process with TF-IDF, giving preference to keyword-rich sentences
    tf_idf_sentences = _process_sentence_chunk(valid_sentences, top_n)
    
    # Make sure we include some keyword-rich sentences even if TF-IDF didn't select them
    tf_idf_indices = [sentences.index(s) for s in tf_idf_sentences]
    
    # Add some keyword-rich sentences if they weren't already selected
    missing_keyword_indices = [idx for idx in keyword_rich_indices[:15] if idx not in tf_idf_indices]
    
    # If we need to add keyword-rich sentences, replace some of the TF-IDF selections
    if missing_keyword_indices and len(tf_idf_sentences) >= top_n:
        # Replace some TF-IDF sentences with keyword-rich ones
        replacement_count = min(len(missing_keyword_indices), int(top_n * 0.3))  # Replace up to 30%
        
        # Keep most of the TF-IDF sentences but replace some with keyword-rich ones
        keep_count = top_n - replacement_count
        final_sentences = tf_idf_sentences[:keep_count]
        
        # Add the keyword-rich sentences
        for idx in missing_keyword_indices[:replacement_count]:
            if 0 <= idx < len(sentences):  # Ensure index is valid
                final_sentences.append(sentences[idx])
        
        return final_sentences
    
    return tf_idf_sentences

def _process_sentence_chunk(sentences, top_n):
    """Helper function to process a chunk of sentences with TF-IDF
    
    Args:
        sentences: List of sentences to process
        top_n: Number of top sentences to return
        
    Returns:
        List of most important sentences from the chunk
    """
    if not sentences:
        return []
        
    preprocessed = [preprocess_text(sentence) for sentence in sentences]
    
    # Use TF-IDF to determine sentence importance
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
        
        # Calculate sentence scores based on term importance
        sentence_scores = [(i, sum(tfidf_matrix[i].toarray()[0])) for i in range(len(sentences))]
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N sentences
        top_indices = [score[0] for score in sentence_scores[:top_n]]
        # Keep original order if we want to preserve document flow
        # Sort indices to maintain original document order
        top_indices.sort() 
        important_sentences = [sentences[i] for i in top_indices]
        
        return important_sentences
    except Exception as e:
        logger.error(f"Error extracting important sentences: {str(e)}")
        # Fallback to returning a subset of sentences if TF-IDF fails
        return sentences[:min(top_n, len(sentences))]

def paraphrase_sentence(sentence):
    """Paraphrase a sentence to make it different from the original text
    
    This function restructures the sentence while preserving its meaning,
    to ensure questions don't match the original document word-for-word.
    
    Args:
        sentence: The original sentence
        
    Returns:
        A paraphrased version of the sentence
    """
    # Simple rule-based paraphrasing
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    
    # Dictionary of synonyms for common words
    synonym_map = {
        'use': ['utilize', 'employ', 'apply'],
        'make': ['create', 'produce', 'generate'],
        'good': ['beneficial', 'positive', 'favorable'],
        'important': ['significant', 'crucial', 'essential'],
        'big': ['large', 'substantial', 'considerable'],
        'small': ['little', 'tiny', 'minor'],
        'say': ['state', 'express', 'mention'],
        'show': ['demonstrate', 'indicate', 'display'],
        'tell': ['inform', 'notify', 'communicate'],
        'get': ['obtain', 'acquire', 'receive'],
        'find': ['discover', 'locate', 'identify'],
        'think': ['believe', 'consider', 'assume'],
    }
    
    # Restructuring patterns
    restructuring_patterns = [
        # Passive to active
        (r'is (.*) by', r'\1s'),
        # Active to passive
        (r'(\w+)s the', r'the is \1ed by'),
        # Change word order
        (r'(.*) is (.*) for (.*)', r'for \3, \1 is \2'),
        # Invert conditional
        (r'if (.*), (.*)', r'\2 when \1'),
    ]
    
    # Attempt structure-level paraphrasing
    paraphrased = sentence
    
    # First try restructuring patterns for major changes
    for pattern, replacement in restructuring_patterns:
        if random.random() < 0.3:  # Apply some patterns randomly
            try:
                new_sentence = re.sub(pattern, replacement, paraphrased)
                if new_sentence != paraphrased and len(new_sentence) > 10:
                    paraphrased = new_sentence
                    break
            except:
                continue
    
    # Then replace some words with synonyms
    new_words = []
    for word, tag in tagged_words:
        if word.lower() in synonym_map and random.random() < 0.4:  # 40% chance to replace
            synonyms = synonym_map[word.lower()]
            new_word = random.choice(synonyms)
            
            # Match case of original word
            if word.isupper():
                new_word = new_word.upper()
            elif word[0].isupper():
                new_word = new_word.capitalize()
                
            new_words.append(new_word)
        else:
            new_words.append(word)
    
    # If we have changes from synonym replacement, use that version
    synonym_paraphrase = ' '.join(new_words)
    if synonym_paraphrase != ' '.join(words):
        paraphrased = synonym_paraphrase
    
    # Add/remove transition words
    transition_additions = [
        "Specifically, ", "Moreover, ", "In fact, ", "Indeed, ", 
        "As a result, ", "Consequently, ", "Therefore, "
    ]
    
    if random.random() < 0.3 and not any(paraphrased.startswith(t.lower()) for t in transition_additions):
        paraphrased = random.choice(transition_additions) + paraphrased[0].lower() + paraphrased[1:]
    
    # Convert between sentence structures
    if random.random() < 0.3 and "," in paraphrased:
        parts = paraphrased.split(",", 1)
        if len(parts) == 2 and len(parts[0]) > 5 and len(parts[1]) > 5:
            paraphrased = parts[1].strip() + " " + parts[0].strip() + "."
    
    # Change sentence endings
    if paraphrased.endswith("."):
        paraphrased = paraphrased[:-1]
    paraphrased += "."
    
    # Return original if paraphrasing failed
    if paraphrased == sentence or len(paraphrased) < 10:
        # At least swap a few words around
        if len(words) > 5:
            mid = len(words) // 2
            if random.random() < 0.5:
                paraphrased = " ".join(words[mid:] + words[:mid])
            else:
                # Just add a clarifying phrase
                paraphrased = sentence + " This is a key concept."
    
    return paraphrased

def generate_mcq(sentence):
    """Generate a multiple choice question from a sentence
    
    This function identifies key concepts in a sentence and creates
    an MCQ by replacing a key term with a blank and generating
    plausible alternative answers.
    
    Args:
        sentence: The sentence to convert into an MCQ
        
    Returns:
        Dictionary with question, options, and correct answer or None if
        no suitable question could be generated
    """
    # First, paraphrase the sentence to make it different from the original
    paraphrased_sentence = paraphrase_sentence(sentence)
    
    # Now process the paraphrased sentence
    words = word_tokenize(paraphrased_sentence)
    tagged_words = pos_tag(words)
    
    # Find nouns, verbs, or adjectives that could be good options for MCQs
    # Prioritize nouns and more complex words as they're usually key concepts
    potential_answers = []
    
    # First pass: Look for multi-word noun phrases (better for key concepts)
    noun_phrase = []
    for i, (word, tag) in enumerate(tagged_words):
        if tag.startswith('NN') and len(word) > 3 and word.lower() not in stopwords.words('english'):
            noun_phrase.append(word)
        elif noun_phrase:
            if len(noun_phrase) > 1:  # Only consider phrases with 2+ words
                potential_answers.append(' '.join(noun_phrase))
            noun_phrase = []
    
    # Add the last noun phrase if we had one at the end
    if noun_phrase and len(noun_phrase) > 1:
        potential_answers.append(' '.join(noun_phrase))
    
    # Second pass: Individual words if we didn't find good phrases
    if not potential_answers:
        for word, tag in tagged_words:
            # Focus on important parts of speech
            if len(word) > 3 and word.lower() not in stopwords.words('english'):
                # Prioritize nouns, then verbs, then adjectives
                if tag.startswith('NN'):  # Nouns
                    potential_answers.append(word)
                elif tag.startswith('VB') and len(word) > 5:  # Longer verbs
                    potential_answers.append(word)
                elif tag.startswith('JJ') and len(word) > 6:  # Longer adjectives
                    potential_answers.append(word)
    
    # If no good answer candidates, try from the original sentence
    if not potential_answers:
        orig_words = word_tokenize(sentence)
        orig_tagged = pos_tag(orig_words)
        for word, tag in orig_tagged:
            if tag.startswith('NN') and len(word) > 3 and word.lower() not in stopwords.words('english'):
                potential_answers.append(word)
    
    if not potential_answers:
        return None
    
    # Weight longer words/phrases higher as they're often more important concepts
    weights = [len(answer.split()) + len(answer) / 10 for answer in potential_answers]
    answer = random.choices(potential_answers, weights=weights, k=1)[0]
    
    # Create the question by replacing the answer with a blank
    # For better readability, add a blank with consistent length
    blank = "___________"
    question_text = paraphrased_sentence.replace(answer, blank)
    
    # If the replacement didn't work (e.g., answer is part of another word), try again
    if question_text == paraphrased_sentence:
        pattern = re.compile(r'\b' + re.escape(answer) + r'\b', re.IGNORECASE)
        question_text = pattern.sub(blank, paraphrased_sentence)
    
    # For clarity, add instructional context
    question = f"Complete the following passage by filling in the blank: \"{question_text}\""
    
    # Generate wrong options - plausible alternatives are important and should be related to the topic
    wrong_options = []
    stop_words = set(stopwords.words('english'))
    
    # Determine if the answer is a noun, verb, or adjective
    answer_type = None
    for word, tag in tagged_words:
        if word.lower() == answer.lower() or (len(answer.split()) > 1 and answer.lower().startswith(word.lower())):
            if tag.startswith('NN'):
                answer_type = 'noun'
            elif tag.startswith('VB'):
                answer_type = 'verb'
            elif tag.startswith('JJ'):
                answer_type = 'adjective'
            break
    
    # Try to find thematically related words in the text rather than just any words
    # Words with same POS and similar context are better distractors
    
    # Phase 1: Look for key terms with similar role in the sentence
    similar_key_terms = []
    
    # For multi-word phrases, try to find other multi-word phrases
    if len(answer.split()) > 1:
        # Extract other noun phrases from the text as alternatives
        current_phrase = []
        for i, (word, tag) in enumerate(tagged_words):
            if tag.startswith('NN') or tag.startswith('JJ'):
                current_phrase.append(word)
            elif current_phrase:
                if len(current_phrase) > 1:  # Multi-word phrases
                    phrase = ' '.join(current_phrase)
                    if phrase.lower() != answer.lower():
                        similar_key_terms.append(phrase)
                current_phrase = []
        
        # Add the last phrase if there was one
        if current_phrase and len(current_phrase) > 1:
            phrase = ' '.join(current_phrase)
            if phrase.lower() != answer.lower():
                similar_key_terms.append(phrase)
    
    # Phase 2: Look for other important terms with same part of speech
    for word, tag in tagged_words:
        if len(word) > 3 and word.lower() not in stop_words:
            # Match the part of speech with the answer for more plausible alternatives
            if (answer_type == 'noun' and tag.startswith('NN')) or \
               (answer_type == 'verb' and tag.startswith('VB')) or \
               (answer_type == 'adjective' and tag.startswith('JJ')):
                if word.lower() != answer.lower() and word not in similar_key_terms:
                    similar_key_terms.append(word)
    
    # Phase 3: Pull some key terms from the whole text
    text_words = [w for w in words if len(w) > 4 and w.lower() not in stop_words and w.lower() != answer.lower()]
    unique_text_words = list(set(text_words))
    
    # Combine all potential wrong options, prioritizing similar key terms
    all_potential_options = similar_key_terms + unique_text_words
    
    # Remove duplicates and the correct answer
    all_potential_options = [opt for opt in all_potential_options if opt.lower() != answer.lower()]
    all_potential_options = list(dict.fromkeys(all_potential_options))  # Remove duplicates while preserving order
    
    # If we have enough options, select the most plausible ones
    if len(all_potential_options) >= 3:
        # Prioritize similar_key_terms first
        if len(similar_key_terms) >= 3:
            wrong_options = random.sample(similar_key_terms, 3)
        else:
            # Take all similar_key_terms and fill the rest from unique_text_words
            wrong_options = similar_key_terms.copy()
            remaining_needed = 3 - len(wrong_options)
            remaining_options = [w for w in unique_text_words if w not in wrong_options]
            
            if len(remaining_options) >= remaining_needed:
                wrong_options.extend(random.sample(remaining_options, remaining_needed))
            else:
                wrong_options.extend(remaining_options)
    else:
        # Use whatever options we found
        wrong_options = all_potential_options.copy()
    
    # If we still don't have enough, try to add related concepts based on the answer
    if len(wrong_options) < 3:
        # For multi-word answers, we can use parts of it
        if len(answer.split()) > 1 and len(wrong_options) < 3:
            parts = answer.split()
            for part in parts:
                if len(part) > 3 and part.lower() not in stop_words:
                    if part not in wrong_options and part.lower() != answer.lower():
                        wrong_options.append(part)
                        if len(wrong_options) >= 3:
                            break
        
        # If we still don't have enough options, use semantic variations
        if len(wrong_options) < 3:
            if answer_type == 'noun':
                semantic_options = ["type of " + answer, answer + "s", "similar " + answer]
            elif answer_type == 'verb':
                semantic_options = [answer + "ing", "ability to " + answer, answer + "ed"]
            elif answer_type == 'adjective':
                semantic_options = [answer + "ly", "more " + answer, "very " + answer]
            else:
                semantic_options = ["variation of " + answer, "similar to " + answer, answer + " (modified)"]
                
            for option in semantic_options:
                if option not in wrong_options and option.lower() != answer.lower():
                    wrong_options.append(option)
                    if len(wrong_options) >= 3:
                        break
    
    # If we're still short on options, add these more generic but related options
    topic_based_options = [
        "A related concept",
        "A similar term",
        "An alternative form"
    ]
    
    while len(wrong_options) < 3:
        if topic_based_options:
            option = topic_based_options.pop(0)
            if option not in wrong_options:
                wrong_options.append(option)
        else:
            # Last resort - use generic options
            generic_options = [
                "None of the above",
                "All of the above",
                "Cannot be determined"
            ]
            option = generic_options.pop(0)
            if option not in wrong_options:
                wrong_options.append(option)
    
    # Ensure we have exactly 3 wrong options
    if len(wrong_options) > 3:
        wrong_options = wrong_options[:3]
    
    # Create a list of all options including the correct answer
    options = wrong_options + [answer]
    random.shuffle(options)
    
    return {
        'question': question,
        'options': options,
        'correct_answer': answer
    }

def generate_subjective_question(sentence):
    """Generate a thought-provoking subjective question from a sentence
    
    This function analyzes the sentence structure and content to create
    appropriate subjective questions that encourage deeper thinking and
    analytical responses.
    
    Args:
        sentence: The sentence to convert into a subjective question
        
    Returns:
        Dictionary with question and model answer
    """
    # First, paraphrase the sentence to make it different from the original
    paraphrased_context = paraphrase_sentence(sentence)
    
    # Different types of subjective questions based on Bloom's taxonomy of learning
    # From lower-order to higher-order thinking skills
    question_templates = {
        'knowledge': [
            "Based on the following text: \"{context}\" Define the term {concept}.",
            "From the passage: \"{context}\" Explain what is meant by {concept}.",
            "According to this text: \"{context}\" Describe the characteristics of {concept}.",
            "Referring to this passage: \"{context}\" What is {concept}?",
            "From this excerpt: \"{context}\" Outline the key features of {concept}.",
        ],
        'comprehension': [
            "After reading this: \"{context}\" Explain in your own words the meaning of {concept}.",
            "Based on this text: \"{context}\" Summarize the key points about {concept}.",
            "From the following: \"{context}\" Describe how {concept} works.",
            "In this context: \"{context}\" Clarify the relationship between {concept} and {related}.",
            "From this passage: \"{context}\" Distinguish between {concept} and {related}.",
        ],
        'application': [
            "After reading: \"{context}\" How would you apply {concept} to solve a problem?",
            "Based on this text: \"{context}\" Demonstrate how {concept} could be used in a real-world scenario.",
            "From this passage: \"{context}\" What examples can you provide to illustrate {concept}?",
            "Given this information: \"{context}\" How would you implement {concept} in practice?",
            "From the following: \"{context}\" Apply the concept of {concept} to explain {related}.",
        ],
        'analysis': [
            "Based on this passage: \"{context}\" Analyze the components of {concept}.",
            "From this text: \"{context}\" What are the underlying assumptions behind {concept}?",
            "After reading this: \"{context}\" Compare and contrast {concept} with {related}.",
            "In this excerpt: \"{context}\" What evidence supports {concept}?",
            "Given the following: \"{context}\" Break down the structure of {concept} into its constituent parts.",
        ],
        'evaluation': [
            "From this passage: \"{context}\" Evaluate the effectiveness of {concept}.",
            "Based on this text: \"{context}\" What are the strengths and weaknesses of {concept}?",
            "After reading: \"{context}\" Justify the importance of {concept} in relation to {related}.",
            "Given this information: \"{context}\" Assess the significance of {concept} in the broader context.",
            "From the following: \"{context}\" Critique the validity of {concept} as presented.",
        ],
        'synthesis': [
            "Based on this passage: \"{context}\" How would you modify {concept} to address {related}?",
            "After reading this text: \"{context}\" Propose an alternative approach to {concept}.",
            "From this information: \"{context}\" Develop a plan to implement {concept} more effectively.",
            "Given the following: \"{context}\" Create a new application for {concept}.",
            "Based on the passage: \"{context}\" How might {concept} evolve in the future?",
        ]
    }
    
    # Try to extract key phrases from both original and paraphrased sentences to get more variety
    words_original = word_tokenize(sentence)
    tagged_original = pos_tag(words_original)
    
    words_paraphrased = word_tokenize(paraphrased_context)
    tagged_paraphrased = pos_tag(words_paraphrased)
    
    # Extract noun phrases (potential key concepts) from both versions
    key_phrases = []
    
    # Process both original and paraphrased to get a wider range of key terms
    for tagged_words in [tagged_original, tagged_paraphrased]:
        current_phrase = []
        for word, tag in tagged_words:
            # Look for noun phrases or important concepts
            if tag.startswith('NN') or tag.startswith('JJ'):
                current_phrase.append(word)
            else:
                if current_phrase:
                    key_phrases.append(' '.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:  # Add the last phrase if any
            key_phrases.append(' '.join(current_phrase))
    
    # Remove duplicates
    key_phrases = list(dict.fromkeys(key_phrases))
    
    # Filter out very short phrases
    key_phrases = [phrase for phrase in key_phrases if len(phrase.split()) >= 1]
    
    # If no key phrases found, try individual important words
    if not key_phrases:
        for tagged_words in [tagged_original, tagged_paraphrased]:
            for word, tag in tagged_words:
                if len(word) > 4 and word.lower() not in stopwords.words('english'):
                    if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
                        key_phrases.append(word)
    
    if not key_phrases:
        # Final fallback if we still didn't find key phrases
        return {
            'question': f"Explain the following concept: '{paraphrased_context}'",
            'answer': sentence
        }
    
    # Weight longer phrases higher as they're usually more specific concepts
    weights = [len(phrase.split()) for phrase in key_phrases]
    main_concept = random.choices(key_phrases, weights=weights, k=1)[0]
    
    # Find a related concept if possible (for comparison questions)
    related_concepts = [phrase for phrase in key_phrases if phrase != main_concept]
    related_concept = random.choice(related_concepts) if related_concepts else ""
    
    # For better context presentation, shorten very long sentences
    context = paraphrased_context
    if len(paraphrased_context) > 150:  # If sentence is very long
        # Try to use the most relevant part of the sentence for context
        main_concept_index = paraphrased_context.lower().find(main_concept.lower())
        if main_concept_index >= 0:
            # Get text around the main concept
            start = max(0, main_concept_index - 50)
            end = min(len(paraphrased_context), main_concept_index + len(main_concept) + 50)
            context = paraphrased_context[start:end]
            # Add ellipsis if we truncated
            if start > 0:
                context = "..." + context
            if end < len(paraphrased_context):
                context = context + "..."
    
    # Based on the complexity and structure of the sentence,
    # choose an appropriate question type
    # Longer sentences with multiple concepts favor higher-order questions
    if len(sentence.split()) > 20 and len(key_phrases) >= 3:
        # Complex sentences - use higher-order questions
        question_types = ['analysis', 'evaluation', 'synthesis']
    elif len(sentence.split()) > 12 or len(key_phrases) >= 2:
        # Moderate complexity - use mid-level questions
        question_types = ['comprehension', 'application', 'analysis']
    else:
        # Simple sentences - use basic questions
        question_types = ['knowledge', 'comprehension']
    
    # Choose a question type and template
    question_type = random.choice(question_types)
    template = random.choice(question_templates[question_type])
    
    # Format the question with the concepts and context
    question = template.format(
        concept=main_concept, 
        related=related_concept, 
        context=context
    )
    
    # Remove any trailing periods for consistency
    question = question.rstrip('.')
    
    # Ensure the question ends with a question mark if it's phrased as a question
    if any(question.startswith(starter) for starter in ["What", "How", "Why", "When", "Where", "Which"]):
        if not question.endswith('?'):
            question += "?"
    else:
        # Otherwise, end with a period for imperative statements
        if not question.endswith('.'):
            question += "."
    
    return {
        'question': question,
        'answer': sentence
    }

def generate_questions(text, min_mcq=10, min_subjective=5, max_questions=100):
    """Generate MCQ and subjective questions from document text
    
    Args:
        text: The document text to analyze
        min_mcq: Minimum number of MCQs to generate
        min_subjective: Minimum number of subjective questions to generate
        max_questions: Maximum total questions to limit processing time
        
    Returns:
        Tuple of (mcq_questions, subjective_questions)
    """
    try:
        # Clean and prepare the text
        cleaned_text = preprocess_text(text)
        
        # Determine optimal number of questions based on document length
        # Roughly 1 question per 100 words, with a minimum and maximum
        word_count = len(cleaned_text.split())
        optimal_question_count = max(min_mcq + min_subjective, min(max_questions, word_count // 100))
        
        # Allocate 70% to MCQs and 30% to subjective questions
        optimal_mcq_count = max(min_mcq, int(optimal_question_count * 0.7))
        optimal_subj_count = max(min_subjective, int(optimal_question_count * 0.3))
        
        logger.info(f"Document has {word_count} words. Generating approximately {optimal_mcq_count} MCQs " +
                    f"and {optimal_subj_count} subjective questions.")
        
        # Extract important sentences from the text - get more sentences for longer docs
        top_n_sentences = max(50, min(200, word_count // 50))  # Between 50-200 sentences
        important_sentences = extract_important_sentences(text, top_n=top_n_sentences)
        
        if not important_sentences:
            logger.warning("No significant sentences found in the document.")
            return [], []
        
        # Generate MCQs
        mcq_questions = []
        random.shuffle(important_sentences)
        
        for sentence in important_sentences:
            if len(mcq_questions) >= optimal_mcq_count:
                break
                
            mcq = generate_mcq(sentence)
            if mcq:
                mcq_questions.append(mcq)
        
        # Generate subjective questions
        subjective_questions = []
        # Use longer sentences for subjective questions
        remaining_sentences = [s for s in important_sentences if len(s.split()) >= 10]
        random.shuffle(remaining_sentences)
        
        for sentence in remaining_sentences:
            if len(subjective_questions) >= optimal_subj_count:
                break
                
            subj_q = generate_subjective_question(sentence)
            subjective_questions.append(subj_q)
        
        # Log and return results
        total_generated = len(mcq_questions) + len(subjective_questions)
        logger.info(f"Successfully generated {len(mcq_questions)} MCQs and {len(subjective_questions)} " +
                   f"subjective questions (total: {total_generated})")
        
        return mcq_questions, subjective_questions
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return [], []
