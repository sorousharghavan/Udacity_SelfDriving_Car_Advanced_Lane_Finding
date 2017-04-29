import numpy as np

#Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.averaging_size = 25
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, fit_coeffs):
        self.current_fit = fit_coeffs
        print("old", self.best_fit)
        if self.best_fit == None:
            self.best_fit = np.array(self.current_fit)
        elif len(self.recent_xfitted) == self.averaging_size:
            del self.recent_xfitted[0]

        self.recent_xfitted.append(fit_coeffs)
        self.best_fit = np.mean( np.array(self.recent_xfitted), axis=0 )
        print("new", self.best_fit)
