def plot_AUROC(fpr, tpr, auc):
  # Plot
  plt.figure(figsize=(15,12))
  roc = plt.plot(fpr,tpr,label='ROC curve', lw=3)
  rnd_roc = plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--', label='Random ROC curve')
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.xlabel('FPR',fontsize=20)
  plt.ylabel('TPR',fontsize=20)
  plt.title('ROC curve, AUC = %.3f'%auc,fontsize=25,pad=10)
  plt.fill_between(fpr, tpr, alpha=0.3)
  # Create empty plot with blank marke