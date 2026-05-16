import sys
sys.path.append('DL/pipeline')
import pandas as pd
from batch_inference import predict_batch

sample_logs = pd.DataFrame([{
    'timestamp': '2015-08-21 11:16:06.804',
    'process_id': None,
    'log_level': 'INFO',
    'component': 'org.apache.hadoop.hdfs.server.datanode.DataNode',
    'content': 'registered UNIX signal handlers for [TERM, HUP]'
}])

print("Testing INFO log with different sensitivities:")
print("="*50)

# Test with 'normal' sensitivity
results_normal = predict_batch(sample_logs, sensitivity='normal')
print("Normal sensitivity: Score={:.6f}, Label={}, Severity={}".format(
    results_normal["anomaly_score"].iloc[0],
    results_normal["label"].iloc[0],
    results_normal["severity"].iloc[0]
))

# Test with 'low' sensitivity  
results_low = predict_batch(sample_logs, sensitivity='low')
print("Low sensitivity: Score={:.6f}, Label={}, Severity={}".format(
    results_low["anomaly_score"].iloc[0],
    results_low["label"].iloc[0],
    results_low["severity"].iloc[0]
))

# Test with 'high' sensitivity
results_high = predict_batch(sample_logs, sensitivity='high')
print("High sensitivity: Score={:.6f}, Label={}, Severity={}".format(
    results_high["anomaly_score"].iloc[0],
    results_high["label"].iloc[0],
    results_high["severity"].iloc[0]
))

print("="*50)
print("Threshold values:")
print("  Normal: {:.6f}".format(0.047261))
print("  Low: {:.6f}".format(0.047261 * 1.5))
print("  High: {:.6f}".format(0.047261 * 0.7))