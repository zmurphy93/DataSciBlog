from model import RidgeClassification, KNearestNeighborsClassification, RadiusNeighborsClassification #L1LogisticRegressionClassification
#L2LogisticRegressionClassification, LinearDiscriminantAnalysisClassification, 
#QuadraticDiscriminantAnalysisClassification, L1LinearSVMClassification,
#L2LinearSVMClassification, KernelSVMClassification, GaussianNBClassification,
#StochasticGradientDescentClassification, GaussiaProcessClassification,
#DecisionTreeClassification, BaggingClassification, RandomForestClassification,
#GradientBoostingClassification, AdaboostClassification, XGBoostClassification,
#PerceptronClassification, MultiLayerPerceptronClassification


from flask import Flask, render_template, request
import sys

app = Flask(__name__)

try:
    template_name = sys.argv[1]
except IndexError:
    template_name = 'view'

if template_name == 'view':
    from flask_bootstrap import Bootstrap
    Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template(template_name+'.html')

@app.route('/algorithm', methods=['GET', 'POST'])
def param_select():
    if request.method=="POST":
        algorithm = request.form.get("algorithms", None)
        if algorithm != None:
            if algorithm == "Ridge":
                form = RidgeClassification(request.form)
                #if request.method == 'POST' and form.validate():
                 #   from compute import RidgeCompute as compute
                  #  result = compute(form.Alpha.data, form.Fit_intercept.data,
                   #      form.Max_iterations.data, form.Tol.data, form.Solver.data)
            elif algorithm == "K-Nearest Neighbors":
                form = KNearestNeighborsClassification(request.form)
            elif algorithm == "Radius Neighbors":
                form = RadiusNeighborsClassification(request.form)
            elif algorithm == "L1 Logistic Regression":
                form = L1LogisticRegressionClassification(request.form)
            elif algorithm == "L2 Logistic Regression":
                form = L2LogisticRegressionClassification(request.form)
            elif algorithm == "Linear Discriminant Analysis":
                form = LinearDiscriminantAnalysisClassification(request.form)
            elif algorithm == "Quadratic Discriminant Analysis":
                form = QuadraticDiscriminantAnalysisClassification(request.form)
            elif algorithm == "L1 LinearSVM":
                form = L1LinearSVMClassification(request.form)
            elif algorithm == "L2 LinearSVM":
                form = L2LinearSVMClassification(request.form)
            elif algorithm == "Support Vector Classification":
                form =KernelSVMClassificationClassification(request.form)
            elif algorithm == "GaussianNB":
                form = GaussianNBClassification(request.form)
            elif algorithm == "Stochastic Gradient Descent":
                form = StochasticGradientDescentClassification(request.form)
            elif algorithm == "Gaussian Processes":
                form = GaussianProcessClassification(request.form)
            elif algorithm == "Decision Tree":
                form = DecisionTreeClassification(request.form)
            elif algorithm == "Bagging":
                form = BaggingClassification(request.form)
            elif algorithm == "Random Forest":
                form = RandomForestClassification(request.form)
            elif algorithm == "AdaBoost":
                form = AdaboostClassification(request.form)
            elif algorithm == "GBRT":
                form = GradientBoostingClassification(request.form)
            elif algorithm == "XGBoost":
                form = XGBoostClassification(request.form)
            elif algorithm == "Perceptron":
                form = PerceptronClassification(request.form)
            elif algorithm == "MLP":
                form = MultiLayerPerceptronClassification(request.form)
            else:
                result = None
            print (form, dir(form))
            for f in form:
               print(f.id)
               print(f.name)
               print(f.label)

            return render_template(template_name+".html", algorithm=algorithm, form=form)
   
#@app.route('/algorithm', methods=['GET', 'POST'])
#def train_Classification(form):
#    form = InputForm(request.form)
#    if request.method == 'POST' and form.validate():
#        result = compute(form.A.data, form.b.data,
#                         form.w.data, form.T.data)
#    else:
#        result = None
#    print (form, dir(form))
#    print (form.keys())
#    for f in form:
#        print(f.id)
#        print(f.name)
#        print(f.label)
#
#    return render_template(template_name + '.html',
#                           form=form, result=result)

if __name__ == '__main__':
    app.run()