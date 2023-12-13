from tqdm import tqdm
import torch
import torch.nn.functional as F

class PrototypeDrifting(object):
    def __init__(self, proto_shift_dict, num_of_prototype=0, PROTO_augmentation_w_COV=False, cov_shift_dict=None, PROTO_sketching=False, sketch_mat=None):
        self.sigma = proto_shift_dict["sigma"]
        self.mean_Balance_beta = proto_shift_dict["mean_Balance_beta"]  # original: alpha
        self.mean_MovingAvg_eta = proto_shift_dict["mean_MovingAvg_eta"] # original: beta
        self.using_delta = proto_shift_dict["using_delta"]
        
        self.PROTO_augmentation_w_COV = PROTO_augmentation_w_COV
        self.cov_Shrinkage_alpha = cov_shift_dict["cov_Shrinkage_alpha"] 
        self.cov_MovingAvg_eta = cov_shift_dict["cov_MovingAvg_eta"] 
        self.cov_Balance_beta = cov_shift_dict["cov_Balance_beta"] 
        self.PROTO_sketching = PROTO_sketching
        self.sketch_mat = sketch_mat
        print('PrototypeDrifting | mean | sigma: {0}, mean_MovingAvg_eta: {1}, mean_Balance_beta: {2}, using_delta: {3}'.format(self.sigma, self.mean_MovingAvg_eta, self.mean_Balance_beta, self.using_delta))
        print('PrototypeDrifting | covariance | cov_Shrinkage_alpha: {0}, cov_MovingAvg_eta: {1}, cov_Balance_beta: {2}'.format(self.cov_Shrinkage_alpha, self.cov_MovingAvg_eta, self.cov_Balance_beta))
        print('PrototypeDrifting | PROTO_sketching: {0}, sketch_mat: {1}'.format(self.PROTO_sketching, self.sketch_mat))
        print('PrototypeDrifting | num_of_prototype: {0}'.format(num_of_prototype))
        
        self.previous_feature_drift = []
        for i in range(num_of_prototype):
            self.previous_feature_drift.append(None)
        self.previous_updated_covariance = []
        for i in range(num_of_prototype):
            self.previous_updated_covariance.append(None)

    def prototype_update(self, cls_feature_before_updating, cls_index_before_updating, cls_feature_after_updating, cls_index_after_updating, class_wise_mean, class_wise_covariance=None):
        if self.using_delta:
            # print('PrototypeDrifting | using delta for prototype updating')
            semantic_drift, _ = self.calculate_semantic_drift_for_current_data(cls_feature_before_updating, cls_index_before_updating, cls_feature_after_updating, cls_index_after_updating)
            updated_prototype, updated_cov = self.assume_semantic_drift_for_previous_cls_prototype_using_delta(semantic_drift, cls_feature_before_updating, class_wise_mean, cls_feature_after_updating, class_wise_covariance)
        else:
            # print('PrototypeDrifting | directly using feature for prototype updating')
            updated_prototype, updated_cov = self.assume_semantic_drift_for_previous_cls_prototype_not_using_delta(cls_feature_after_updating, cls_feature_before_updating, class_wise_mean)
        
        return updated_prototype, updated_cov

    def calculate_semantic_drift_for_current_data(self, cls_feature_before_updating, cls_index_before_updating, cls_feature_after_updating, cls_index_after_updating):
        semantic_drift_list = []
        cls_index_list = []

        for i in range(len(cls_feature_before_updating)):
            # print('TRIPLET_DIST_W_PROTO | i: {0}'.format(i))
            # print('TRIPLET_DIST_W_PROTO | cls_feature_before_updating[{0}][:5]: {0}'.format(i, cls_feature_before_updating[i][:5]))
            if cls_index_before_updating[i] != cls_index_after_updating[i]:
                raise ValueError('Something is wrong')
            current_semantic_drift = cls_feature_after_updating[i] - cls_feature_before_updating[i]
            semantic_drift_list.append(current_semantic_drift)
            cls_index_list.append(cls_index_before_updating[i])
    
        semantic_drift = torch.stack(semantic_drift_list, dim=0)
        cls_index = torch.stack(cls_index_list, dim=0)

        if self.PROTO_sketching:
            semantic_drift=semantic_drift.type(torch.cuda.DoubleTensor)
            semantic_drift = semantic_drift.mm(self.sketch_mat)

        return semantic_drift, cls_index

    def assume_semantic_drift_for_previous_cls_prototype_using_delta(self, current_data_semantic_drift, cls_feature_before_updating, class_wise_mean, cls_feature_after_updating=None, class_wise_covariance=None):
        # using delta
        
        updated_class_wise_mean_list = []
        updated_class_wise_convariance_list = []

        if self.PROTO_sketching:
            cls_feature_before_updating = cls_feature_before_updating.type(torch.cuda.DoubleTensor)
            cls_feature_before_updating = cls_feature_before_updating.mm(self.sketch_mat)


        for i in range(len(class_wise_mean)):
            cls_feature_prototype = class_wise_mean[i].expand(cls_feature_before_updating.size())
            distance = F.pairwise_distance(cls_feature_before_updating, cls_feature_prototype, p=2)
    
            distance = torch.square(distance)
            divider = 2 * (self.sigma ** 2)
            omega = torch.exp(-(distance/divider)).view(-1,1)

            # cls_feature_prototype_norm = torch.norm(cls_feature_prototype)
            # cls_feature_before_updating_norm = torch.norm(cls_feature_before_updating)
            # omega_norm = torch.norm(omega)
            # print('cls_feature_prototype_norm: {0}, cls_feature_before_updating_norm: {1}, omega_norm: {2}'.format(cls_feature_prototype_norm, cls_feature_before_updating_norm, omega_norm))
    
            cls_feature_drift_denominator = torch.sum(omega*current_data_semantic_drift, dim=0)
            cls_feature_drift_numerator = torch.sum(omega, dim=0).clamp(1e-12)
            cls_feature_drift = cls_feature_drift_denominator / cls_feature_drift_numerator
        
            if self.mean_MovingAvg_eta > 0:
                if self.previous_feature_drift[i] == None:
                    final_cls_feature_drift = cls_feature_drift
                    self.previous_feature_drift[i] = final_cls_feature_drift
                else:
                    final_cls_feature_drift = self.mean_MovingAvg_eta * self.previous_feature_drift[i] + (1-self.mean_MovingAvg_eta) * cls_feature_drift
                    self.previous_feature_drift[i] = final_cls_feature_drift
                updated_cls_feature_prototype = torch.add(class_wise_mean[i], self.mean_Balance_beta*final_cls_feature_drift)
                updated_class_wise_mean_list.append(updated_cls_feature_prototype)
            else:
                updated_cls_feature_prototype = torch.add(class_wise_mean[i], self.mean_Balance_beta*cls_feature_drift)
                updated_class_wise_mean_list.append(updated_cls_feature_prototype)

            if self.PROTO_augmentation_w_COV:
                difference = cls_feature_after_updating - updated_cls_feature_prototype
                difference = torch.reshape(difference, (difference.size()[0], difference.size()[1], 1))
                difference_matrix = difference * torch.transpose(difference, 1, 2)
                    
                omega = torch.reshape(omega, (omega.size()[0], 1, 1))
                update_covariance_denominator = torch.sum(omega*difference_matrix, dim=0)
                update_covariance_numerator = torch.sum(omega, dim=0).clamp(1e-12)
                update_covariance = update_covariance_denominator / update_covariance_numerator


                nFeatS=class_wise_covariance[i].size()[0]
                mfBatchDiagOne=torch.ones(nFeatS).diag().type(class_wise_covariance.type()).cuda(0)
                shrinkaged_update_covariance = (1 - self.cov_Shrinkage_alpha) * update_covariance + self.cov_Shrinkage_alpha * mfBatchDiagOne

                if self.previous_updated_covariance[i] == None:
                    final_updated_covariance = shrinkaged_update_covariance
                    self.previous_updated_covariance[i] = final_updated_covariance
                else:
                    final_updated_covariance = self.cov_MovingAvg_eta * self.previous_updated_covariance[i] + (1-self.cov_MovingAvg_eta) * shrinkaged_update_covariance
                    self.previous_updated_covariance[i] = final_updated_covariance

                balanced_final_updated_covariance = (1 - self.cov_Balance_beta) * class_wise_covariance[i] + self.cov_Balance_beta * final_updated_covariance 
            else:
                balanced_final_updated_covariance = None
            updated_class_wise_convariance_list.append(balanced_final_updated_covariance)

        mean_tensor = torch.stack(updated_class_wise_mean_list)
        if self.PROTO_augmentation_w_COV:
            convariance_matrix_tensor = torch.stack(updated_class_wise_convariance_list)
        else:
            convariance_matrix_tensor = None

        # print('mean_tensor: {0}'.format(mean_tensor.size()))
        # print('convariance_matrix_tensor: {0}'.format(convariance_matrix_tensor.size()))
        return mean_tensor, convariance_matrix_tensor


    def assume_semantic_drift_for_previous_cls_prototype_not_using_delta(self, current_feature_after_updating, cls_feature_before_updating, class_wise_mean):
        # not using delta
        
        updated_class_wise_mean_list = []

        for i in range(len(class_wise_mean)):
            cls_feature_prototype = class_wise_mean[i].expand(cls_feature_before_updating.size())
            distance = F.pairwise_distance(cls_feature_before_updating, cls_feature_prototype, p=2)
    
            distance = torch.square(distance)
            divider = 2 * (self.sigma ** 2)
            omega = torch.exp(-(distance/divider)).view(-1,1)
            # print('omega: {0}'.format(omega))
    
            cls_feature_drift_denominator = torch.sum(omega*current_feature_after_updating, dim=0)
            cls_feature_drift_numerator = torch.sum(omega, dim=0).clamp(1e-12)
            cls_feature_drift = cls_feature_drift_denominator / cls_feature_drift_numerator

            if self.using_gamma_moving_avg:
                if self.previous_feature_drift[i] == None:
                    final_cls_feature_drift = cls_feature_drift
                    self.previous_feature_drift[i] = final_cls_feature_drift
                else:
                    final_cls_feature_drift = self.gamma * self.previous_feature_drift[i] + (1-self.gamma) * cls_feature_drift
                    self.previous_feature_drift[i] = final_cls_feature_drift
                updated_cls_feature_prototype = torch.add((1-self.mean_Balance_beta)*class_wise_mean[i], self.mean_Balance_beta*final_cls_feature_drift)
                updated_class_wise_mean_list.append(updated_cls_feature_prototype)
            else:
                updated_cls_feature_prototype = torch.add((1-self.mean_Balance_beta)*class_wise_mean[i], self.mean_Balance_beta*cls_feature_drift)
                updated_class_wise_mean_list.append(updated_cls_feature_prototype)                

        ret = torch.stack(updated_class_wise_mean_list)
        return ret


