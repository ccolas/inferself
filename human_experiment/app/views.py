from flask import render_template, request, make_response
from . import app
from .models import db, Subject, Trial

study_route = '/'

@app.route('/logic', methods=['GET'])
def logic():
    return render_template('logic_game.html', game_type="logic")

@app.route('/logic_click', methods=['GET'])
def logic_click():
    return render_template('logic_game.html', game_type="logic_click")

@app.route('/logic_u', methods=['GET'])
def logic_u():
    return render_template('logic_game.html', game_type="logic_u")

@app.route('/contingency', methods=['GET'])
def contingency():
    return render_template('contingency_game.html', game_type="contingency")

@app.route('/contingency_u', methods=['GET'])
def contingency_u():
    return render_template('contingency_game.html', game_type="contingency_u")

@app.route('/contingency_2', methods=['GET'])
def contingency_2():
    return render_template('contingency_game.html', game_type="contingency_2")
@app.route('/contingency_6', methods=['GET'])
def contingency_6():
    return render_template('contingency_game.html', game_type="contingency_6")
@app.route('/contingency_8', methods=['GET'])
def contingency_8():
    return render_template('contingency_game.html', game_type="contingency_8")
@app.route('/contingency_noisy', methods=['GET'])
def contingency_noisy():
    return render_template('contingency_game.html', game_type="contingency_noisy")

@app.route('/shuffle_keys', methods=['GET'])
def shuffle_keys():
    return render_template('contingency_game.html', game_type="shuffle_keys")

@app.route('/shuffle_keys_u', methods=['GET'])
def shuffle_keys_u():
    return render_template('contingency_game.html', game_type="shuffle_keys_u")

@app.route('/change_agent', methods=['GET'])
def change_agent():
    return render_template('contingency_game.html', game_type="change_agent")
@app.route('/change_agent_10', methods=['GET'])
def change_agent_10():
    return render_template('contingency_game.html', game_type="change_agent_10")

@app.route('/change_agent_u', methods=['GET'])
def change_agent_u():
    return render_template('contingency_game.html', game_type="change_agent_u")

#Views
@app.route(study_route, methods=['GET', 'POST'])
def experiment():
    if request.method == 'GET':
        return render_template('experiment.html')
    if request.method == 'POST':
        data = request.get_json(force=True)['data']
        #subject information
        if data['exp_phase'] == 'subject_info':
            print('recording subject data')
            ret = Subject( subject_id= str(data['subject_id']),
                           game_type = str(data['game_type']),
                           vg_experience = str(data['vg_experience']),
                           completion_code = str(data['completion_code']),
                           att1= str(data['att1']),
                           att2= str(data['att2']),
                           comp1= str(data['comp1']),
                           comp2= str(data['comp2']),
                           age= str(data['age']),
                           gender= str(data['gender']),
                           nationality= str(data['nationality']),
                           country= str(data['country']),
                           student= str(data['student']),
                           language= str(data['language']),
                           education= str(data['education']))
            
        #trial response
        else:
            print('recording trial data')

            ret = Trial( row_id = str(data['row_id']),
                         subject_id = str(data['subject_id']),
                         game_data = str(data['game_data']),
                         full_rt = str(data['full_rt']),
                         game_type = str(data['game_type']))

        db.session.add(ret)
        db.session.commit()
        return make_response("", 200)