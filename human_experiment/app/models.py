from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Subject(db.Model):
    __tablename__ = 'subjects'
    
    subject_id = db.Column(db.String, primary_key=True)
    game_type = db.Column(db.String)
    completion_code = db.Column(db.String)
    age = db.Column(db.String)
    att1= db.Column(db.String)
    att2= db.Column(db.String)
    comp1= db.Column(db.String)
    comp2= db.Column(db.String)
    gender = db.Column(db.String)
    nationality = db.Column(db.String)
    country = db.Column(db.String)
    student = db.Column(db.String)
    language = db.Column(db.String)
    education = db.Column(db.String)
    vg_experience = db.Column(db.String)

    def __repr__(self):
        return '<Subject %r>' % self.id

class Trial(db.Model):
    __tablename__ = 'trials'
    row_id = db.Column(db.String, primary_key=True)
    subject_id = db.Column(db.String)
    game_data = db.Column(db.String)
    full_rt = db.Column(db.String)
    game_type = db.Column(db.String)
    
    def __repr__(self):
        return '<Subject %r>' % self.id