import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
from models import Document, Question, db
from text_processor import extract_text_from_pdf, extract_text_from_txt
from question_generator import generate_questions

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-for-assessment-app")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///assessment.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize the database
db.init_app(app)

# Add a context processor to provide current date to templates
@app.context_processor
def inject_now():
    return {'now': datetime.now}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    documents = Document.query.all()
    return render_template('index.html', documents=documents)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.referrer)
    
    file = request.files['file']
    title = request.form.get('title', '')
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.referrer)
    
    if file and allowed_file(file.filename):
        try:
            # Create unique filename
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            ext = filename.rsplit('.', 1)[1].lower()
            safe_filename = f"{unique_id}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            
            # Save file
            file.save(filepath)
            
            # Extract text from file
            if ext == 'pdf':
                text_content = extract_text_from_pdf(filepath)
            else:
                text_content = extract_text_from_txt(filepath)
                
            # Create document in DB
            document = Document(
                title=title if title else filename,
                filename=safe_filename,
                original_filename=filename,
                content=text_content
            )
            db.session.add(document)
            db.session.commit()
            
            # Generate questions in the background
            generate_and_save_questions(document.id)
            
            flash('File uploaded successfully and being processed', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            flash(f'Error uploading file: {str(e)}', 'danger')
            return redirect(request.referrer)
    else:
        flash('File type not allowed. Please upload PDF or TXT files.', 'warning')
        return redirect(request.referrer)

def generate_and_save_questions(document_id):
    """Generate questions for a document and save them to the database
    
    This function analyzes the document content and generates an optimal number
    of questions based on the document length and complexity.
    """
    try:
        document = Document.query.get(document_id)
        if not document:
            logger.error(f"Document with ID {document_id} not found")
            return
            
        # Generate questions - dynamically based on document size
        content = document.content
        
        # Set reasonable limits based on document length:
        # - Min 10 MCQs and 5 subjective questions
        # - Max 100 questions total (to limit processing time)
        # The function will determine optimal numbers based on text length
        mcq_questions, subjective_questions = generate_questions(
            content, 
            min_mcq=10, 
            min_subjective=5, 
            max_questions=100
        )
        
        # Save MCQ questions
        for q in mcq_questions:
            question = Question(
                document_id=document_id,
                question_text=q['question'],
                question_type='mcq',
                options=",".join(q['options']),
                correct_answer=q['correct_answer']
            )
            db.session.add(question)
        
        # Save subjective questions
        for q in subjective_questions:
            question = Question(
                document_id=document_id,
                question_text=q['question'],
                question_type='subjective',
                correct_answer=q['answer']
            )
            db.session.add(question)
            
        db.session.commit()
        document.processed = True
        db.session.commit()
        
        total_questions = len(mcq_questions) + len(subjective_questions)
        logger.info(f"Successfully generated and saved {total_questions} questions for document {document_id}")
    
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        db.session.rollback()

@app.route('/practice/<int:document_id>')
def practice(document_id):
    document = Document.query.get_or_404(document_id)
    mcq_questions = Question.query.filter_by(document_id=document_id, question_type='mcq').all()
    subjective_questions = Question.query.filter_by(document_id=document_id, question_type='subjective').all()
    
    return render_template('practice.html', 
                          document=document, 
                          mcq_questions=mcq_questions, 
                          subjective_questions=subjective_questions)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<int:document_id>', methods=['POST'])
def delete_document(document_id):
    document = Document.query.get_or_404(document_id)
    
    # Delete associated questions
    Question.query.filter_by(document_id=document_id).delete()
    
    # Delete the file from the filesystem
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
    
    # Delete the document record
    db.session.delete(document)
    db.session.commit()
    
    flash('Document and associated questions deleted successfully', 'success')
    return redirect(url_for('index'))

# Create database tables within app context
with app.app_context():
    db.create_all()
