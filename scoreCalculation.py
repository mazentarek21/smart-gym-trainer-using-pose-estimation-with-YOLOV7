def score_stage_up(angle,curr_score):

    # Define the minimum and maximum angles in the acceptable range
    acceptable_min = 30
    acceptable_max = 45
    
    # Calculate the score based on a linear mapping between the angle and the score scale
    if angle < acceptable_min:
        score = 0
    elif angle > acceptable_max:
        score = curr_score + 10
    else:
        score = (angle - acceptable_min) / (acceptable_max - acceptable_min) * 10
        score = score + curr_score
        
    return score

def score_stage_down(angle,curr_score):

    # Define the minimum and maximum angles in the acceptable range
    acceptable_min = 150
    acceptable_max = 180
    
    # Calculate the score based on a linear mapping between the angle and the score scale
    if angle < acceptable_min:
        score = 0
    elif angle > acceptable_max:
        score = curr_score + 10
    else:
        score = (angle - acceptable_min) / (acceptable_max - acceptable_min) * 10
        score = score + curr_score
        
    return score

def score_evaluation(stage_up_score, stage_down_score):
    return ((stage_up_score + stage_down_score)/2)