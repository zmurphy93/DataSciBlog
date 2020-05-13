from wtforms import Form, FloatField, validators, BooleanField, IntegerField, SelectField, SubmitField

class RidgeClassification(Form):
    Alpha = FloatField(label='Alpha', default=1.0, validators=[validators.InputRequired()])
    Max_iterations = IntegerField(label='Iteration_Number', default=1000, validators=[validators.InputRequired()])
    Tol = FloatField(label='Tolerance Threshold', default=0.001,validators=[validators.InputRequired()])
    Solver = SelectField(u'Solver', choices=[('auto', 'Auto'), ('lsqr', 'Least Squares'),('sparse_cg', 'Sparse Conj. Grad.'),('sag', 'SAG'),('saga', 'SAGA'),('svd', 'Sing. Vec. Decomp.'), ('cholesky', 'Cholesky')])
    Fit_intercept = BooleanField('Use Intercept?')
    Submit = SubmitField(label='Train Classifier')       
           
class KNearestNeighborsClassification(Form):
    Neighbors = IntegerField(label='Number of Neighbors', default=5,validators=[validators.InputRequired()])
    LeafSize = IntegerField( label='Leaf Size', default=30,  validators=[validators.InputRequired()])
    P = IntegerField(label='Minkowski parameter', default=1, validators=[validators.InputRequired()])
    Algorithm = SelectField(u'Algorithm', choices=[('auto', 'Auto'), ('ball_tree', 'Ball Tree'),('kd_tree', 'KD-Tree'),('brute', 'Brute Force Search')])
    Weights = SelectField(u'Weights', choices=[('uniform', 'Uniform'), ('distance', 'Distance')])
    Metric = SelectField(u'Distance Metric', choices=[('euclidean', 'Euclidean'), ('manhattan', 'Manhattan'),('minkowski', 'Minkowski.'),('chebyshev', 'Chebyshev')])
    Submit = SubmitField(label='Train Classifier') 
   
class RadiusNeighborsClassification(Form):
    Radius = IntegerField(label='Radius', default=5,validators=[validators.InputRequired()])
    LeafSize = IntegerField( label='Leaf Size', default=30,  validators=[validators.InputRequired()])
    P = IntegerField(label='Minkowski parameter', default=1, validators=[validators.InputRequired()])
    Algorithm = SelectField(u'Algorithm', choices=[('auto', 'Auto'), ('ball_tree', 'Ball Tree'),('kd_tree', 'KD-Tree'),('brute', 'Brute Force Search')])
    Weights = SelectField(u'Weights', choices=[('uniform', 'Uniform'), ('distance', 'Distance')])
    Metric = SelectField(u'Distance Metric', choices=[('euclidean', 'Euclidean'), ('manhattan', 'Manhattan'),('minkowski', 'Minkowski.'),('chebyshev', 'Chebyshev')])
    Submit = SubmitField(label='Train Classifier') 

    
#class DecisionTreeClassification(Form):
#    Criterion = FloatField(
#        label='Impurity Metric', default=1.0,
#        validators=[validators.InputRequired()])
#    Splitter = FloatField(
#        label='Splitting Scheme', default=0,
#        validators=[validators.InputRequired()])
#    Max_depth = FloatField(
#        label='Depth of the Tree', default=2*pi,
#        validators=[validators.InputRequired()])
#    Min_samples_split = FloatField(
#        label='Minimum Samples Split', default=18,
#        validators=[validators.InputRequired()])
#    Min_samples_leaf = FloatField(
#        label='Minimum Samples per Leaf', default=18,
#        validators=[validators.InputRequired()])
#    Min_weight_fraction_leaf = FloatField(
#        label='Minimum Weight Fraction', default=18,
#        validators=[validators.InputRequired()])
#    Max_features = FloatField(
#        label='Maximum number of features', default=18,
#        validators=[validators.InputRequired()])
#    Max_leaf_nodes = FloatField(
#        label='Number of Leaf Nodes', default=18,
#        validators=[validators.InputRequired()])
#    Min_impurity_split = FloatField(
#        label='Minimum Impurity Split', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class L1LinearSVMClassification(Form):
#    Loss = FloatField(
#        label='Loss function', default=0,
#        validators=[validators.InputRequired()])
#    Dual = FloatField(
#        label='Use Dual Formulation', default=2*pi,
#        validators=[validators.InputRequired()])
#    Tol = FloatField(
#        label='Tolerance for Stopping', default=18,
#        validators=[validators.InputRequired()])
#    C = FloatField(
#        label='Penalization', default=18,
#        validators=[validators.InputRequired()])
#    Fit_intercept = FloatField(
#        label='Use intercept', default=18,
#        validators=[validators.InputRequired()])
#    Max_iter = FloatField(
#        label='Number of iterations', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class L2LinearSVMClassification(Form):
#    Loss = FloatField(
#        label='Loss function', default=0,
#        validators=[validators.InputRequired()])
#    Dual = FloatField(
#        label='Use Dual Formulation', default=2*pi,
#        validators=[validators.InputRequired()])
#    Tol = FloatField(
#        label='Tolerance for Stopping', default=18,
#        validators=[validators.InputRequired()])
#    C = FloatField(
#        label='Penalization', default=18,
#        validators=[validators.InputRequired()])
#    Fit_intercept = FloatField(
#        label='Use intercept', default=18,
#        validators=[validators.InputRequired()])
#    Max_iter = FloatField(
#        label='Number of iterations', default=18,
#        validators=[validators.InputRequired()])
#   
#class KernelSVMClassification(Form):
#    C = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    Kernel = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    Degree = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    Gamma = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Coef0 = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Shrinking = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Tol = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Max_iter = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class GaussianNBClassification(Form):
#    Var_smoothing = FloatField(
#        label='Variance Smoothing', default=1.0,
#        validators=[validators.InputRequired()])
#   
#   
#   
#class StochasticGradientDescentClassification(Form):
#    A = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    b = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    w = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    T = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class GaussianProcessClassification(Form):
#    A = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    b = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    w = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    T = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class RandomForestClassification(Form):
#    N_estimators= FloatField(
#        label='Number of Trees', default=1.0,
#        validators=[validators.InputRequired()])
#    Criterion = FloatField(
#        label='Impurity Metric', default=1.0,
#        validators=[validators.InputRequired()])
#    Splitter = FloatField(
#        label='Splitting Scheme', default=0,
#        validators=[validators.InputRequired()])
#    Max_depth = FloatField(
#        label='Depth of the Tree', default=2*pi,
#        validators=[validators.InputRequired()])
#    Min_samples_split = FloatField(
#        label='Minimum Samples Split', default=18,
#        validators=[validators.InputRequired()])
#    Min_samples_leaf = FloatField(
#        label='Minimum Samples per Leaf', default=18,
#        validators=[validators.InputRequired()])
#    Min_weight_fraction_leaf = FloatField(
#        label='Minimum Weight Fraction', default=18,
#        validators=[validators.InputRequired()])
#    Max_features = FloatField(
#        label='Maximum number of features', default=18,
#        validators=[validators.InputRequired()])
#    Max_leaf_nodes = FloatField(
#        label='Number of Leaf Nodes', default=18,
#        validators=[validators.InputRequired()])
#    Min_impurity_split = FloatField(
#        label='Minimum Impurity Split', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class BaggingClassification(Form):
#    Base_estimator = FloatField(
#        label='Algorithm', default=1.0,
#        validators=[validators.InputRequired()])
#    N_estimators = FloatField(
#        label='Number of Algorithms', default=0,
#        validators=[validators.InputRequired()])
#    Max_samples = FloatField(
#        label='Number of Samples', default=2*pi,
#        validators=[validators.InputRequired()])
#    Bootstrap = FloatField(
#        label='Use Bootstrap', default=2*pi,
#        validators=[validators.InputRequired()])
#   
#   
class AdaboostClassification(Form):
    Base_estimator = Algorithm = SelectField(u'Classifier', choices=[('auto', 'Auto'), ('ball_tree', 'Ball Tree'),('kd_tree', 'KD-Tree'),('brute', 'Brute Force Search')])
    N_estimators = IntegerField(label='Number of Classifiers', default=50,validators=[validators.InputRequired()])
    Learning_rate = FloatField(label='Learning Rate', default=0.001,validators=[validators.InputRequired()])
#   
#   
#class GradientBoostingClassification(Form):
#    A = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    b = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    w = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    T = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class XGBoostClassification(Form):
#    A = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    b = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    w = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    T = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class PerceptronClassification(Form):
#    A = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    b = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    w = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    T = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#   
#   
#class MultiLayerPerceptronClassification(Form):
#    Hidden_layer_size = FloatField(
#        label='amplitude (m)', default=1.0,
#        validators=[validators.InputRequired()])
#    Activation = FloatField(
#        label='damping factor (kg/s)', default=0,
#        validators=[validators.InputRequired()])
#    Solver = FloatField(
#        label='frequency (1/s)', default=2*pi,
#        validators=[validators.InputRequired()])
#    Alpha = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Batch_size = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Learning_rate = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Learning_rate_init = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Power_t = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Max_iter = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Tol = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Momentum = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Nesterovs_momentum = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Early_stopping = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Validation_fraction = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Beta_1 = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Beta_2 = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    Epsilon = FloatField(
#        label='time interval (s)', default=18,
#        validators=[validators.InputRequired()])
#    