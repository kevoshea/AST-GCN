from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import helper as h
import os

# protocol 1
def calculate_point_wise(ground_truth, prediction):
    """
    Calculates precision, recall, and F1-score for anomaly detection.

    Args:
      ground_truth (array-like): Ground truth labels of anomalies.
      prediction (array-like): Predicted labels of anomalies.

    Returns:
      tuple: A tuple of (precision, recall, F1-score), all rounded to two decimal places.
    """

    precision = precision_score(ground_truth, prediction, average='binary')
    recall = recall_score(ground_truth, prediction, average='binary')
    f1 = f1_score(ground_truth, prediction, average='binary')

    return round(precision, 2), round(recall, 2), round(f1, 2)

# protocol 2
def calculate_point_adjusted(ground_truth, prediction):
    # Ensure inputs are flattened to 1D arrays
    ground_truth = np.array(ground_truth).flatten()
    prediction = np.array(prediction).flatten()

    # Create DataFrame
    df = pd.DataFrame({'anomaly': ground_truth, 'result': prediction})
    df['group'] = (df['anomaly'] != df['anomaly'].shift()).cumsum()

    # Identify anomaly and prediction groups
    anomaly_groups = df[df['anomaly'] == 1]['group'].unique()
    prediction_groups = df[df['result'] == 1]['group'].unique()
    valid_groups = set(anomaly_groups) & set(prediction_groups)

    # Adjust predictions
    df['adjusted_result'] = df.apply(lambda x: 1 if x['group'] in valid_groups or x['result'] == 1 else 0, axis=1)

    # Calculate metrics
    precision = precision_score(df['anomaly'], df['adjusted_result'])
    recall = recall_score(df['anomaly'], df['adjusted_result'])
    f1 = f1_score(df['anomaly'], df['adjusted_result'])

    return round(precision, 2), round(recall, 2), round(f1, 2)

# protocol 3
def calculate_composite(ground_truth, prediction):
    """
    Calculates the Composite F1 Score using point-level precision and event-level recall.
    """
    precision = precision_score(ground_truth, prediction)
    
    ground_truth_events = identify_events(ground_truth)
    prediction_events = identify_events(prediction)
    
    recall = calculate_event_level_recall(ground_truth_events, prediction_events)
    
    if precision + recall == 0:  # avoid division by zero
        f1c = 0
    else:
        f1c = 2 * (precision * recall) / (precision + recall)
    return round(precision, 2), round(recall, 2), round(f1c, 2)

# protocol 4
def calculate_event_wise(ground_truth, prediction):
    """
    Calculate event-wise Precision, Recall, and F1 score.
    """
    # identify events in ground truth and prediction
    gt_events = identify_events(ground_truth)
    pred_events = identify_events(prediction)
    
    # calculate TPE and FPE
    tpe = len([gt for gt in gt_events if any(max(p[0], gt[0]) <= min(p[1], gt[1]) for p in pred_events)])
    fpe = len([p for p in pred_events if not any(max(p[0], gt[0]) <= min(p[1], gt[1]) for gt in gt_events)])
    
    # calculate event-wise Recall
    recall_event_wise = tpe / len(gt_events) if gt_events else 0
    
    # calculate FAR
    normal_points = ground_truth[ground_truth == 0]
    # false_positives = len([p for p in prediction.index if prediction[p] == 1 and ground_truth[p] == 0])
    false_positives = sum(1 for pred, truth in zip(prediction, ground_truth) if pred == 1 and truth == 0)
    far = calculate_far(normal_points, false_positives)

    # calculate event-wise Precision
    precision_event_wise = (tpe / (tpe + fpe)) * (1 - far) if tpe + fpe > 0 else 0
    
    # calculate event-wise F1 Score
    if precision_event_wise + recall_event_wise == 0:  # Avoid division by zero
        f1_event_wise = 0
    else:
        f1_event_wise = 2 * (precision_event_wise * recall_event_wise) / (precision_event_wise + recall_event_wise)
    
    return round(precision_event_wise, 2), round(recall_event_wise, 2), round(f1_event_wise, 2)

# evaluate using the 4 protocols and return pandas df
def evaluate(ground_truth, prediction):
    df_methods = pd.DataFrame({'Point-wise': calculate_point_wise(ground_truth, prediction),
                 'Point-adjust': calculate_point_adjusted(ground_truth, prediction),
                 'Composite': calculate_composite(ground_truth, prediction),
                 'Eventwise': calculate_event_wise(ground_truth, prediction)})
    df_methods = df_methods.apply(lambda x: round(x, 2)).T
    df_methods.columns = ['Precision', 'Recall', 'F1']
    return df_methods

def composite_protocol_evaluation(ground_truth, prediction):
    """
    Calculates composite evaluation metrics: event-based recall, point-based precision, combined F1-score.

    Args:
      ground_truth (array-like): Ground truth labels.
      prediction (array-like):  Predicted labels

    Returns:
      tuple: Event-based recall, point-based precision, combined F1-score.
    """

    # Grouping
    df = pd.DataFrame({'anomaly': ground_truth, 'result': prediction})
    df['group'] = (df['anomaly'] != df['anomaly'].shift()).cumsum()

    # Calculate Event-Based Metrics
    events = set(df['group'].unique())
    prediction_events = set(df[df['result'] == 1]['group'].unique())
    anomaly_events = set(df[df['anomaly'] == 1]['group'].unique())

    events_tp = len(prediction_events & anomaly_events)
    events_fp = len(prediction_events - anomaly_events)
    events_fn = len(anomaly_events - prediction_events)  

    r_events = events_tp / (events_tp + events_fn)

    # Calculate Point-Based Precision
    p_p = precision_score(ground_truth, prediction, average='binary')  

    # Combined F1-Score
    f1_c = 2 * (r_events * p_p) / (r_events + p_p)

    return r_events, p_p, round(f1_c, 2)


def calculate_event_level_recall(ground_truth_events, prediction_events):
    """
    Calculates the recall at the event level.
    """
    true_positive_events = 0
    for gt_event in ground_truth_events:
        for pred_event in prediction_events:
            if gt_event[0] <= pred_event[0] <= gt_event[1] or gt_event[0] <= pred_event[1] <= gt_event[1]:
                true_positive_events += 1
                break
    recall = true_positive_events / len(ground_truth_events) if ground_truth_events else 0
    return recall

def is_within_interval(time, intervals):
    for start, end in intervals:
        if start <= time <= end:
            return True 
    return False  

def aggregate_cw_ad(files):
    cw_ad = [h.process_CW_metrics_file(f, columns_to_concat=['TopicName']) for f in files]
    cw_ad_df = cw_ad[0] 

    for i in range(1, len(cw_ad)):
        cw_ad_df = pd.merge(cw_ad_df, cw_ad[i], right_index=True, left_index=True, how='outer')

    cw_ad_df['cw_pred'] = cw_ad_df.fillna(0).max(axis=1).astype(int)
    return cw_ad_df[['cw_pred']]

def calculate_far(normal_points, false_positives):
    """
    Calculate the False Alarm Rate (FAR).
    """
    if len(normal_points) == 0:  # Avoid division by zero
        return 0
    return false_positives / len(normal_points)

def identify_events(series):
    """
    Identifies anomalous events in a binary series and returns the start and end indices for each event.
    """
    events = []
    in_event = False
    for i, value in enumerate(series):
        if value == 1 and not in_event:
            start = i
            in_event = True
        elif value == 0 and in_event:
            end = i
            events.append((start, end))
            in_event = False
    # Handle case where the last event goes until the end of the series
    if in_event:
        events.append((start, len(series)))
    return events

# def composite_protocol_evaluation(df):
#     # recall is event-based
#     # precision is point-based
#     df['group'] = (df['anomaly'] != df['anomaly'].shift()).cumsum()
#     events = set(df['group'].unique())
#     prediction_events = set(df[df['result']==1]['group'].unique())
#     anomaly_events = set(df[df['anomaly']==1]['group'].unique())
#     events_tp = len(prediction_events & anomaly_events)
#     events_fp = len(prediction_events - anomaly_events)
#     events_tn = len((events-anomaly_events)&(events-prediction_events))
#     events_fn = len((events-anomaly_events)-(events-prediction_events))
    
#     r_events = events_tp/(events_tp+events_fn)
#     p_p, r_p, _ = point_wise_evaluation(df)
#     f1_c = 2*(r_events*r_p)/(r_events+r_p)
    
#     return r_events, p_p, round(f1_c, 2)

# def point_adjusted_evaluation(df):
#     df['group'] = (df['anomaly'] != df['anomaly'].shift()).cumsum()
#     anomaly_groups = df[df['anomaly'] == 1]['group'].unique()
#     prediction_groups = df[df['result'] == 1]['group'].unique()
#     valid_groups = set(anomaly_groups) & set(prediction_groups)
#     df['adjusted_result'] = df.apply(lambda x: 1 if x['group'] in valid_groups or x['result']==1 else 0, axis=1)
    
#     precision = precision_score(df['anomaly'], df['adjusted_result'])
#     recall = recall_score(df['anomaly'], df['adjusted_result'])
#     f1 = f1_score(df['anomaly'], df['adjusted_result'])
    
#     return round(precision, 2), round(recall, 2), round(f1, 2)

# def point_wise_evaluation(df):
#     precision, recall, f1_score, _ = precision_recall_fscore_support(df['anomaly'], df['result'], average='binary')
#     return round(precision, 2), round(recall, 2), round(f1_score, 2)
