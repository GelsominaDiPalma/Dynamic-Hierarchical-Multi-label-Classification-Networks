import scikit_posthocs as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

dict_data = {
    'DHN': [0.256, 0.198, 0.307, 0.301, 0.285, 0.260, 0.295, 0.216, 0.415, 0.371, 0.456, 0.448, 0.440, 0.416, 0.447, 0.382, 0.768, 0.764, 0.958, 0.928,
            0.7137,0.9025,0.91,0.916,0.739,0.8576,0.9042,0.8431,0.8361,0.8961],
    'C-HMCNN': [0.255, 0.195, 0.306, 0.302, 0.286, 0.258, 0.292, 0.215, 0.413, 0.370, 0.455, 0.447, 0.436, 0.414, 0.446, 0.382, 0.758, 0.756, 0.956, 0.927,
                0.7168,0.9012,0.909,0.9112,0.7319,0.8583,0.904,0.8426,0.8366,0.8937],
    'C-HMCNN-Min': [0.249, 0.195, 0.301, 0.294, 0.280, 0.251, 0.282, 0.214, 0.406, 0.368, 0.450, 0.385, 0.430, 0.371, 0.378, 0.376, 0.728, 0.752, 0.951, 0.928,
                0.7014,0.8964,0.8996,0.9068,0.72,0.8454,0.8929,0.8341,0.8236,0.8873],
    'HMC-LMLP': [0.207, 0.182, 0.245, 0.242, 0.235, 0.211, 0.236, 0.186, 0.361, 0.343, 0.406, 0.373, 0.380, 0.371, 0.370, 0.342, 0, 0, 0, 0,
                 0,0,0,0,0,0,0.627,0.671,0.730,0.663], 
    'HMCN-R': [0.247, 0.189, 0.298, 0.300, 0.283, 0.249, 0.290, 0.210, 0.395, 0.368, 0.435, 0.450, 0.416, 0.463, 0.443, 0.375, 0.514, 0.710, 0.904, 0.897,
               0,0,0,0,0,0,0,0,0,0],
    'HMCN-F': [0.252, 0.193, 0.298, 0.301, 0.284, 0.254, 0.291, 0.211, 0.400, 0.369, 0.440, 0.452, 0.428, 0.465, 0.447, 0.376, 0.530, 0.724,  0.950,  0.920,
               0,0,0,0,0,0,0,0,0,0],
    'CLUS-ENS': [0.227, 0.187, 0.286, 0.271, 0.267, 0.231, 0.284, 0.211, 0.387,  0.361,  0.433, 0.422, 0.415, 0.395, 0.438, 0.371, 0.501, 0.696, 0.803, 0.881,
                 0,0.5576,0.7262,0.7918,0.5766,0.6531,0.713,0.693,0.695,0.734],
}
print(stats.friedmanchisquare(*dict_data.values()))

data = (
  pd.DataFrame(dict_data)
  .rename_axis('dataset')
  .melt(
      var_name='estimator',
      value_name='score',
      ignore_index=False,
  )
  .reset_index()
)
data['block_id'] = data['dataset']
test_results = sp.posthoc_nemenyi_friedman(
   data,
   melted=True,
    block_col='dataset',
    group_col='estimator',
   y_col='score',
   block_id_col='block_id', 
)

# plt.title('Heatmap of p values')
sp.sign_plot(test_results)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.savefig("nemenyi_p.pdf", format='pdf', bbox_inches='tight')

avg_rank = data.groupby('dataset').score.rank(pct=False, ascending=False).groupby(data.estimator).mean()

print(avg_rank)

print(avg_rank.values)


plt.figure(figsize=(15, 4), dpi=100)
plt.title('Critical difference diagram of average score ranks')

sp.critical_difference_diagram(
    ranks=avg_rank,
    sig_matrix=test_results,
    label_fmt_left='{label}',         
    label_fmt_right='{label}',        
    text_h_margin=0.3,
    label_props={'fontweight': 'bold', 'fontsize': 11},
    crossbar_props={'marker': 'o', 'markersize': 6},  
    elbow_props={},                                
)
plt.savefig("nemenyi_cd.pdf", format='pdf', bbox_inches='tight')
