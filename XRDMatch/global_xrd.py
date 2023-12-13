class global_var:
    w_noise_ratio = 0
    w_noise_peak = 0.1 
    w_move_gap = 150   
    
def set_value(a,b,c,w_noise_ratio=0.0,w_noise_peak = 0.1,w_move_gap = 250):
    global_var.w_noise_ratio = w_noise_ratio + 0.01*a
    global_var.w_noise_peak = w_noise_peak+2*b
    global_var.w_move_gap = w_move_gap+50*c	    

def get_value():
    return global_var.w_noise_ratio,global_var.w_noise_peak,global_var.w_move_gap

