import tensorflow as tf

import numpy as np


def np_softmax(x, axis=None):
    """
    Softmax as in scipy source code
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

def threshold_for_each_TPR_value(values):
    """
    Creation of the array with the threshold needed to obtain each TPR value
    :values: softmax o logits values in a 1D array
    """
    # Sorted from lower to greater values
    sorted_values = np.sort(values)
    # Inverse the order to get it correctly (greater the threshold, lower the TPR)
    tpr_range = np.arange(0,1,0.01)[::-1]
    tpr_range[0] = 0.99999999 # For selecting the first item correctly ()
    thresholds = np.zeros(len(tpr_range)) # One threshold for each tpr
    for index, tpr in enumerate(tpr_range):
        # El valor del threshold para el valor de TPR que se esta computando en una iteracion es el valor del array
        # de valores ordenados que se encuentra en la posicion X, siendo X el valor del TPR por la longitud del array.
        # Es decir, si estamos con el TPR = 15% (0.15) y la longitud el array es 1000, entonces el valor del threshold
        # que proporciona dicho TPR es el que esta en la posicion 150 (0.15 * 1000) en el array ordenado de los valores.
        thresholds[index] = sorted_values[int(len(sorted_values)*tpr)]
    return thresholds

def compare_likelihood_to_likelihood_thr_one_for_all_classes(distances_evaluating, thr_distances_array):
  '''
  Function that creates an array of shape (tpr, InD_or_OD), where tpr has the lenght of the number of steps of the TPR list
  and second dimensions has the total lenght of the distances_evaluating, and contains True if its InD and False if is ood
  :distances_evaluating: list with each element being an array with the distances to avg clusters of one class [array(.), array(.)]
  :thr_distances_array: array containing the distance for the TPR
   corresponding to that position. For example, the TPR = 0.85 corresponds to the 85th position.
  '''
  in_or_out_distribution_per_tpr = np.zeros((len(thr_distances_array), len(distances_evaluating)), dtype=bool)
  for tpr_index, thr_for_one_tpr in enumerate(thr_distances_array):
      in_or_out_distribution_per_tpr[tpr_index] = np.where(distances_evaluating > thr_for_one_tpr, True, False)

  return in_or_out_distribution_per_tpr

def ind_or_ood_decision(thr, values):
    return np.where(values > thr, 1, 0)

def likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(likelihood_test, likelihood_ood, likelihood_thresholds):
  # Creation of the array with True if predicted InD (True) or OD (False)
  in_or_out_distribution_per_tpr_test = compare_likelihood_to_likelihood_thr_one_for_all_classes(likelihood_test, likelihood_thresholds)
  in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),dtype=bool) # To fix that one element is True when TPR is 0
  in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),dtype=bool) # To fix that last element is True when TPR is 1
  in_or_out_distribution_per_tpr_ood = compare_likelihood_to_likelihood_thr_one_for_all_classes(likelihood_ood, likelihood_thresholds)

  # Creation of arrays with TP, FN and FP, TN
  tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
  fp_tn_ood = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_ood)

  # Computing TPR, FPR and Precision
  tpr_values = tp_fn_test[:,0] / (tp_fn_test[:,0] + tp_fn_test[:,1])
  fpr_values = fp_tn_ood[:,0] / (fp_tn_ood[:,0] + fp_tn_ood[:,1])
  #precision  = tp_fn_test[:,0] / (tp_fn_test[:,0] + fp_tn_ood[:,0])

  # Eliminating NaN value at TPR = 1 and other NaN values that may appear due to
  # precision = TP / (TP + FN) = 0 / (0 + 0)
  #precision[0] = 1
  #print(precision)
  #np.nan_to_num(precision, nan=1, copy=False)


  return tpr_values, fpr_values

def tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr):
  '''
  Function that creates an array with the number of values of tp and fp or fn and tn, depending on if the
  passed array is InD or OD.
  :in_or_out_distribution_per_tpr: array with True if predicted InD and False if predicted OD, for each TPR
  ::return: array with shape (tpr, 2) with the 2 dimensions being tp,fn if passed array is InD, and fp and tn
    if the passed array is OD
  '''
  tp_fn_fp_tn = np.zeros((len(in_or_out_distribution_per_tpr),2),dtype='uint16')
  length_array = in_or_out_distribution_per_tpr.shape[1]
  for index, element in enumerate(in_or_out_distribution_per_tpr):
    n_True = int(len(element.nonzero()[0]))
    tp_fn_fp_tn[index,0] = n_True
    tp_fn_fp_tn[index,1] = length_array - n_True
  return tp_fn_fp_tn

# FROM THE SOURCE CODE OF SCYPY
def np_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """
    FROM THE SOURCE CODE OF SCYPY
    Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
        .. versionadded:: 0.12.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
        .. versionadded:: 0.16.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2
    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.
    Examples
    --------
    """
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

def ood_detection(test_logits, ood_test_logits, type_ood="baseline"):
    #test_logits, ood_test_logits = get_correct_shape(test_logits, ood_test_logits)

    if type_ood == "baseline":
        # Compute the softmax values for each class of each sample
        # and save only the maximum of each example (only the softmax value of the predicted class)
        test_softmax_winners = np.max(np_softmax(test_logits, axis=1), axis=1)
        ood_test_softmax_winners = np.max(np_softmax(ood_test_logits, axis=1), axis=1)
    elif type_ood == "odin":
        temperature = 1000
        # Compute the softmax values for each class of each sample
        # and save only the maximum of each example (only the softmax value of the predicted class)
        test_softmax_winners = np.max(np_softmax(test_logits / temperature, axis=1), axis=1)
        ood_test_softmax_winners = np.max(np_softmax(ood_test_logits / temperature, axis=1), axis=1)
    else:
        temperature = 1
        # Compute the softmax values for each class of each sample
        # and save only the maximum of each example (only the softmax value of the predicted class)
        test_softmax_winners = -(-temperature * np_logsumexp(test_logits / temperature, axis=1))
        ood_test_softmax_winners = -(-temperature * np_logsumexp(ood_test_logits / temperature, axis=1))

    #print("Hasta aqui bien")
    #print(test_logits.shape)
    #print(ood_test_logits.shape)

    # Creation of the array with the thresholds for each TPR (True Positive Rate) value. TPR is the number of
    # InD instances that are correctly classified divided by the total number of InD instances.
    thresholds = threshold_for_each_TPR_value(test_softmax_winners)

    # Conmputing precision, tpr (true positive rate) and fpr (false positive rate)
    tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(test_softmax_winners, ood_test_softmax_winners, thresholds)
    # Appending that when FPR = 1 the TPR is also 1, for AUROC computation:
    tpr_values_auroc = np.append(tpr_values, 1)
    fpr_values_auroc = np.append(fpr_values, 1)
    # AUROC
    auroc = np.trapz(tpr_values_auroc, fpr_values_auroc)
    # AUPR
    # results.append([temp, auroc, aupr])
    #print(type_ood)
    print('-' * 30)
    print(f'AUROC = {auroc * 100:.3f} %')
    print('-' * 30, '\n')

    # Vector con la decision para cada instancia, asignando 1 si se decide que es InD y 0 si es OoD
    # Lo dejo aqui el codigo por si te quieres quedar solo con el vector de las decisiones
    # decided_tpr_value = 80 # in percentage
    # train_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], train_softmax_winners)
    # test_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], test_softmax_winners)
    # ood_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], ood_test_softmax_winners)

    return auroc



    #if __name__ == '__main__':

    # ____ OoD detection ____
    # El input para cada método son los logits de train, test y OoD
    # El calculo del threshold está hecho cogiendo todos los valores de train, pero puede hacerse cogiendo solo una
    # parte o directamente con los de test o como se prefiera



    # _____________ Baseline _____________ 
    
    # Compute the softmax values for each class of each sample 
    # and save only the maximum of each example (only the softmax value of the predicted class)    
    #train_softmax_winners = np.max(np_softmax(train_logits, axis=1), axis=1)
    #test_softmax_winners = np.max(np_softmax(test_logits, axis=1), axis=1)
    #ood_test_softmax_winners = np.max(np_softmax(ood_test_logits, axis=1), axis=1)

    # Creation of the array with the thresholds for each TPR (True Positive Rate) value. TPR is the number of
    # InD instances that are correctly classified divided by the total number of InD instances.
    #thresholds = threshold_for_each_TPR_value(train_softmax_winners)
    # Conmputing precision, tpr (true positive rate) and fpr (false positive rate)
    #precision, tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
    #    test_softmax_winners, ood_test_softmax_winners, thresholds)
    # Appending that when FPR = 1 the TPR is also 1, for AUROC computation:
    #tpr_values_auroc = np.append(tpr_values, 1)
    #fpr_values_auroc = np.append(fpr_values, 1)
    # AUROC
    #auroc = np.trapz(tpr_values_auroc, fpr_values_auroc)
    # AUPR
    #aupr = np.trapz(precision, tpr_values)
    # results.append([temp, auroc, aupr])
    #print('Baseline')
    #print('-' * 30)
    #print(f'AUROC = {auroc * 100:.3f} %')
    #print(f'AUPR  = {aupr * 100:.3f} %')
    #print('-' * 30, '\n')

    # Vector con la decision para cada instancia, asignando 1 si se decide que es InD y 0 si es OoD
    # Lo dejo aqui el codigo por si te quieres quedar solo con el vector de las decisiones
    #decided_tpr_value = 80 # in percentage
    #train_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], train_softmax_winners)
    #test_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], test_softmax_winners)
    #ood_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], ood_test_softmax_winners)


    # _____________ ODIN _____________
    # Hay que decidir el parámetro de la temperatura
    #temperature = 1000

    # Compute the softmax values for each class of each sample
    # and save only the maximum of each example (only the softmax value of the predicted class)
    #train_softmax_winners = np.max(np_softmax(train_logits/temperature, axis=1), axis=1)
    #test_softmax_winners = np.max(np_softmax(test_logits/temperature, axis=1), axis=1)
    #ood_test_softmax_winners = np.max(np_softmax(ood_test_logits/temperature, axis=1), axis=1)

    # Creation of the array with the thresholds for each TPR (True Positive Rate) value. TPR is the number of
    # InD instances that are correctly classified divided by the total number of InD instances.
    #thresholds = threshold_for_each_TPR_value(train_softmax_winners)
    # Conmputing precision, tpr (true positive rate) and fpr (false positive rate)
    #precision, tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
    #    test_softmax_winners, ood_test_softmax_winners, thresholds)
    # Appending that when FPR = 1 the TPR is also 1, for AUROC computation:
    #tpr_values_auroc = np.append(tpr_values, 1)
    #fpr_values_auroc = np.append(fpr_values, 1)
    # AUROC
    #auroc = np.trapz(tpr_values_auroc, fpr_values_auroc)
    # AUPR
    #aupr = np.trapz(precision, tpr_values)
    # results.append([temp, auroc, aupr])
    #print()
    #print('ODIN')
    #print('-' * 30)
    #print(f'AUROC = {auroc * 100:.3f} %')
    #print(f'AUPR  = {aupr * 100:.3f} %')
    #print('-' * 30, '\n')

    # Vector con la decision para cada instancia, asignando 1 si se decide que es InD y 0 si es OoD
    # Lo dejo aqui el codigo por si te quieres quedar solo con el vector de las decisiones
    #decided_tpr_value = 80 # in percentage
    #train_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], train_softmax_winners)
    #test_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], test_softmax_winners)
    #ood_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], ood_test_softmax_winners)

    # _____________ Energy based OoD _____________
    # Hay que decidir el parámetro de la temperatura, pero en el paper lo dejan a 1 porque dicen que es lo que mejor
    # funciona siempre, luego no lo cambiaria
    #temperature = 1

    # Compute the softmax values for each class of each sample
    # and save only the maximum of each example (only the softmax value of the predicted class)
    #energy_train = -(-temperature * np_logsumexp(train_logits / temperature, axis=1))
    #energy_test = -(-temperature * np_logsumexp(test_logits / temperature, axis=1))
    #energy_ood = -(-temperature * np_logsumexp(ood_test_logits / temperature, axis=1))

    # Creation of the array with the thresholds for each TPR (True Positive Rate) value. TPR is the number of
    # InD instances that are correctly classified divided by the total number of InD instances.
    #thresholds = threshold_for_each_TPR_value(energy_train)
    # Conmputing precision, tpr (true positive rate) and fpr (false positive rate)
    #precision, tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
    #    energy_test, energy_ood, thresholds)
    # Appending that when FPR = 1 the TPR is also 1, for AUROC computation:
    #tpr_values_auroc = np.append(tpr_values, 1)
    #fpr_values_auroc = np.append(fpr_values, 1)
    # AUROC
    #auroc = np.trapz(tpr_values_auroc, fpr_values_auroc)
    # AUPR
    #aupr = np.trapz(precision, tpr_values)
    # results.append([temp, auroc, aupr])
    #print()
    #print('Energy OoD')
    #print('-' * 30)
    #print(f'AUROC = {auroc * 100:.3f} %')
    #print(f'AUPR  = {aupr * 100:.3f} %')
    #print('-' * 30, '\n')

    # Vector con la decision para cada instancia, asignando 1 si se decide que es InD y 0 si es OoD
    # Lo dejo aqui el codigo por si te quieres quedar solo con el vector de las decisiones
    #decided_tpr_value = 80 # in percentage
    #train_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], energy_train)
    #test_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], energy_test)
    #ood_decisions = ind_or_ood_decision(thresholds[decided_tpr_value], energy_ood)
