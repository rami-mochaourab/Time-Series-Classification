from .utils.gs_utils import generate_inside_ball, get_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state
import numpy as np



class GrowingSpheres:
    """
    class to fit the Original Growing Spheres algorithm
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=1,
                caps=None,
                n_in_layer=5000, #5000
                first_radius=5, #5
                dicrease_radius=10,
                sparse=True):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))

        if target_class == None:
            target_class = 1 - self.y_obs #works only if binary classification; else specify a target class

        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius
        self.sparse = sparse

        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")


    def find_counterfactual(self):
        ennemies_ = self.exploration()
        
        closest_ennemy_ = sorted(ennemies_,
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0]
        if self.sparse == True:
            out = self.feature_selection(closest_ennemy_)
        else:
            out = closest_ennemy_
        return out


    def exploration(self):
        n_ennemies_ = 999 #999
        radius_ = self.first_radius


        while n_ennemies_ > 0:
            first_layer_ = self.ennemies_in_layer((0, radius_), self.caps, self.n_in_layer)
            n_ennemies_ = first_layer_.shape[0]

            radius_ = radius_ / self.dicrease_radius
            print("%d ennemies found in initial sphere. Zooming in..."%n_ennemies_)
            #print(radius_)
        else:
            print("Exploring...")
            iteration = 0
            step_ = (self.dicrease_radius - 1) * radius_/5 #5.0

            while n_ennemies_ <= 0:

                layer = self.ennemies_in_layer((radius_, radius_ + step_), self.caps, self.n_in_layer)

                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                #print(radius_)
                iteration += 1

            print("Final number of iterations: ", iteration)
        print("Final radius: ", (radius_ - step_, radius_))
        print("Final number of ennemies: ", n_ennemies_)
        return layer


    def ennemies_in_layer(self, segment, caps=None, n=1000):
        """
        prend obs, genere couche dans segment, et renvoie les ennemis dedans
        """
        layer = generate_inside_ball(self.obs_to_interprete, segment, n)

        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)

        preds_ = self.prediction_fn(layer)
        return layer[np.where(preds_ == self.target_class)]


    def feature_selection(self, counterfactual):
        """
        """
        print("Feature selection...")
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete)), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        out = counterfactual.copy()
        reduced = 0

        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete[k]
            if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                out[k] = new_enn[k]
                reduced += 1

        print("Reduced %d coordinates"%reduced)
        return out
