from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with questions
    questions = db.relationship('Question', backref='document', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Document {self.title}>'

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), nullable=False)  # 'mcq' or 'subjective'
    options = db.Column(db.Text, nullable=True)  # For MCQs, comma-separated
    correct_answer = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Question {self.id} ({self.question_type})>'
    
    def get_options(self):
        """Returns the options as a list if this is an MCQ"""
        if self.options:
            return self.options.split(',')
        return []
