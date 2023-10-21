from main import multiRun


# Set Log Path and Device
log_home = './experiment/hurricane'
device_idx = 0

# Set Hyperparameters
ema_momentum = 0.9
lr = 1e-4
weight_u_loss = 20
psl_threshold_h = 0.9
u_loss ='L2'
print(ema_momentum, lr, weight_u_loss, psl_threshold_h, u_loss)

n_labeled_per_class_list = [3, 5, 10, 20]

for n_labeled_per_class in n_labeled_per_class_list:

    # Investigate on debiasing
    # - MemoryBank
    multiRun(device_idx=device_idx, log_home=log_home, investigate='psl_num', psl_mode=True, aug_mode=False, ema_mode=True, ema_momentum=ema_momentum, lr=lr, weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, u_loss=u_loss, n_labeled_per_class=n_labeled_per_class)    
    
    # - Baseline: PSL
    multiRun(device_idx=device_idx, log_home=log_home, psl_mode=True, aug_mode=False, ema_mode=True, ema_momentum=ema_momentum, lr=lr, weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, u_loss=u_loss, n_labeled_per_class=n_labeled_per_class)    
    # - LogitAdjust
    multiRun(device_idx=device_idx, log_home=log_home, threshold_mode='debias', psl_mode=True, aug_mode=False, ema_mode=True, ema_momentum=ema_momentum, lr=lr, weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, u_loss=u_loss, n_labeled_per_class=n_labeled_per_class)    
    # - SAT (Self-Adapter-Thresholding)
    multiRun(device_idx=device_idx, log_home=log_home, threshold_mode='sat', psl_mode=True, aug_mode=False, ema_mode=True, ema_momentum=ema_momentum, lr=lr, weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, u_loss=u_loss, n_labeled_per_class=n_labeled_per_class)    