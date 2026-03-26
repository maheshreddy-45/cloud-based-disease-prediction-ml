from flask import Flask, redirect, render_template, request, url_for
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



app = Flask(__name__)

HOME_PAGE_OVERVIEW = {
    'title': 'Machine Learning and Deep Learning-Based Medical Screening Web App',
    'description': (
        'This web app uses machine learning and deep learning models trained on '
        'structured health data and medical images to estimate disease risk scores.'
    ),
    'disclaimer': (
        'These predictions are intended for screening support only and are not a '
        'substitute for professional medical diagnosis.'
    ),
}

MODEL_ACCURACIES = [
    {'name': 'Diabetes Model', 'accuracy': '98.25%'},
    {'name': 'Breast Cancer Model', 'accuracy': '98.25%'},
    {'name': 'Heart Disease Model', 'accuracy': '85.25%'},
    {'name': 'Kidney Disease Model', 'accuracy': '99%'},
    {'name': 'Liver Disease Model', 'accuracy': '78%'},
    {'name': 'Malaria Model', 'accuracy': '96%'},
    {'name': 'Pneumonia Model', 'accuracy': '95%'},
]

DIABETES_FEATURES = [
    {'key': 'pregnancies', 'label': 'Number of pregnancies', 'kind': 'ordinal', 'step': 1, 'min': 0},
    {'key': 'glucose', 'label': 'Glucose', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'bloodpressure', 'label': 'Blood pressure', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'skinthickness', 'label': 'Skin thickness', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'insulin', 'label': 'Insulin level', 'kind': 'continuous', 'step': 15, 'min': 0},
    {'key': 'bmi', 'label': 'BMI', 'kind': 'continuous', 'step': 1, 'min': 0},
    {'key': 'dpf', 'label': 'Diabetes pedigree function', 'kind': 'continuous', 'step': 0.05, 'min': 0},
    {'key': 'age', 'label': 'Age', 'kind': 'continuous', 'step': 5, 'min': 0},
]

BREAST_CANCER_FEATURES = [
    {'key': 'radius_mean', 'label': 'Radius (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'texture_mean', 'label': 'Texture (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'perimeter_mean', 'label': 'Perimeter (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'area_mean', 'label': 'Area (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'smoothness_mean', 'label': 'Smoothness (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'compactness_mean', 'label': 'Compactness (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'concavity_mean', 'label': 'Concavity (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'concave_points_mean', 'label': 'Concave points (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'symmetry_mean', 'label': 'Symmetry (mean)', 'kind': 'continuous', 'min': 0},
    {'key': 'radius_se', 'label': 'Radius (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'perimeter_se', 'label': 'Perimeter (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'area_se', 'label': 'Area (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'compactness_se', 'label': 'Compactness (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'concavity_se', 'label': 'Concavity (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'concave_points_se', 'label': 'Concave points (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'fractal_dimension_se', 'label': 'Fractal dimension (SE)', 'kind': 'continuous', 'min': 0},
    {'key': 'radius_worst', 'label': 'Radius (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'texture_worst', 'label': 'Texture (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'perimeter_worst', 'label': 'Perimeter (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'area_worst', 'label': 'Area (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'smoothness_worst', 'label': 'Smoothness (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'compactness_worst', 'label': 'Compactness (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'concavity_worst', 'label': 'Concavity (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'concave_points_worst', 'label': 'Concave points (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'symmetry_worst', 'label': 'Symmetry (worst)', 'kind': 'continuous', 'min': 0},
    {'key': 'fractal_dimension_worst', 'label': 'Fractal dimension (worst)', 'kind': 'continuous', 'min': 0},
]

HEART_FEATURES = [
    {'key': 'age', 'label': 'Age', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'sex', 'label': 'Sex', 'kind': 'binary'},
    {'key': 'cp', 'label': 'Chest pain type', 'kind': 'choice', 'choices': [0, 1, 2, 3]},
    {'key': 'trestbps', 'label': 'Resting blood pressure', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'chol', 'label': 'Cholesterol', 'kind': 'continuous', 'step': 15, 'min': 0},
    {'key': 'fbs', 'label': 'Fasting blood sugar', 'kind': 'binary'},
    {'key': 'restecg', 'label': 'Resting ECG result', 'kind': 'choice', 'choices': [0, 1, 2]},
    {'key': 'thalach', 'label': 'Maximum heart rate', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'exang', 'label': 'Exercise-induced angina', 'kind': 'binary'},
    {'key': 'oldpeak', 'label': 'ST depression', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'slope', 'label': 'ST slope', 'kind': 'choice', 'choices': [0, 1, 2]},
    {'key': 'ca', 'label': 'Major vessels by fluoroscopy', 'kind': 'choice', 'choices': [0, 1, 2, 3]},
    {'key': 'thal', 'label': 'Thalassemia test result', 'kind': 'choice', 'choices': [3, 6, 7]},
]

KIDNEY_FEATURES = [
    {'key': 'age', 'label': 'Age', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'bp', 'label': 'Blood pressure', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'al', 'label': 'Albumin', 'kind': 'choice', 'choices': [0, 1, 2, 3, 4, 5]},
    {'key': 'su', 'label': 'Sugar', 'kind': 'choice', 'choices': [0, 1, 2, 3, 4, 5]},
    {'key': 'rbc', 'label': 'Red blood cells', 'kind': 'binary'},
    {'key': 'pc', 'label': 'Pus cell', 'kind': 'binary'},
    {'key': 'pcc', 'label': 'Pus cell clumps', 'kind': 'binary'},
    {'key': 'ba', 'label': 'Bacteria', 'kind': 'binary'},
    {'key': 'bgr', 'label': 'Random blood glucose', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'bu', 'label': 'Blood urea', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'sc', 'label': 'Serum creatinine', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'pot', 'label': 'Potassium', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'wc', 'label': 'White blood cell count', 'kind': 'continuous', 'step': 500, 'min': 0},
    {'key': 'htn', 'label': 'Hypertension', 'kind': 'binary'},
    {'key': 'dm', 'label': 'Diabetes mellitus', 'kind': 'binary'},
    {'key': 'cad', 'label': 'Coronary artery disease', 'kind': 'binary'},
    {'key': 'pe', 'label': 'Pedal edema', 'kind': 'binary'},
    {'key': 'ane', 'label': 'Anemia', 'kind': 'binary'},
]

LIVER_FEATURES = [
    {'key': 'Age', 'label': 'Age', 'kind': 'continuous', 'step': 5, 'min': 0},
    {'key': 'Total_Bilirubin', 'label': 'Total bilirubin', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'Direct_Bilirubin', 'label': 'Direct bilirubin', 'kind': 'continuous', 'step': 0.2, 'min': 0},
    {'key': 'Alkaline_Phosphotase', 'label': 'Alkaline phosphatase', 'kind': 'continuous', 'step': 20, 'min': 0},
    {'key': 'Alamine_Aminotransferase', 'label': 'Alanine aminotransferase', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'Aspartate_Aminotransferase', 'label': 'Aspartate aminotransferase', 'kind': 'continuous', 'step': 10, 'min': 0},
    {'key': 'Total_Protiens', 'label': 'Total proteins', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'Albumin', 'label': 'Albumin', 'kind': 'continuous', 'step': 0.5, 'min': 0},
    {'key': 'Albumin_and_Globulin_Ratio', 'label': 'Albumin and globulin ratio', 'kind': 'continuous', 'step': 0.1, 'min': 0},
    {'key': 'Gender_Male', 'label': 'Male gender flag', 'kind': 'binary'},
]

DISEASE_DETAILS = [
    {
        'name': 'Diabetes',
        'description': (
            'Diabetes is a condition in which blood glucose levels become too high. '
            'Insulin helps glucose move from the blood into the body\'s cells for energy. '
            'When the body does not make enough insulin or cannot use it properly, '
            'glucose remains in the bloodstream.'
        ),
        'symptoms': [
            'Frequent urination',
            'Increased thirst',
            'Extreme fatigue',
            'Blurred vision',
        ],
    },
    {
        'name': 'Breast Cancer',
        'description': (
            'Breast cancer forms in the cells of the breast. It can affect both women '
            'and men, although it is far more common in women.'
        ),
        'symptoms': [
            'A breast lump or thickening that feels different from nearby tissue',
            'Changes in the size, shape, or appearance of a breast',
            'Skin dimpling over the breast',
            'Redness or pitting of the skin over the breast',
        ],
    },
    {
        'name': 'Heart Disease',
        'description': (
            'Heart disease includes conditions that affect the heart and blood vessels. '
            'A common form is coronary artery disease, in which narrowed arteries '
            'reduce blood flow to the heart.'
        ),
        'symptoms': [
            'Chest pain or pressure',
            'Shortness of breath',
            'Fatigue with activity',
            'Pain in the neck, jaw, throat, back, or arm',
        ],
    },
    {
        'name': 'Chronic Kidney Disease',
        'description': (
            'Chronic kidney disease is the gradual loss of kidney function. As kidney '
            'function declines, waste products and excess fluid can build up in the body.'
        ),
        'symptoms': [
            'Nausea',
            'Vomiting',
            'Fatigue and weakness',
            'Muscle cramps',
        ],
    },
    {
        'name': 'Liver Disease',
        'description': (
            'Liver disease includes conditions that damage the liver and affect how it '
            'functions. Some liver diseases cause no symptoms in early stages.'
        ),
        'symptoms': [
            'Swelling of the abdomen or legs',
            'Bruising easily',
            'Changes in stool or urine color',
            'Yellowing of the skin or eyes',
        ],
    },
    {
        'name': 'Malaria',
        'description': (
            'Malaria is a mosquito-borne infectious disease caused by parasites. Symptoms '
            'usually begin several days after the bite of an infected mosquito.'
        ),
        'symptoms': [
            'Fever',
            'Chills',
            'Headache',
            'Nausea and vomiting',
        ],
    },
    {
        'name': 'Pneumonia',
        'description': (
            'Pneumonia is an infection that inflames the air sacs in one or both lungs. '
            'The air sacs may fill with fluid or pus, which can make breathing difficult.'
        ),
        'symptoms': [
            'Cough that may produce mucus',
            'Fever and chills',
            'Shortness of breath',
            'Rapid, shallow breathing',
        ],
    },
]

TABULAR_MODEL_CONFIG = {
    8: {
        'path': 'models/diabetes.pkl',
        'disease_name': 'Diabetes',
        'score_label': 'Diabetes Risk Score',
        'features': DIABETES_FEATURES,
    },
    26: {
        'path': 'models/breast_cancer.pkl',
        'disease_name': 'Breast Cancer',
        'score_label': 'Breast Cancer Risk Score',
        'features': BREAST_CANCER_FEATURES,
    },
    13: {
        'path': 'models/heart.pkl',
        'disease_name': 'Heart Disease',
        'score_label': 'Heart Disease Risk Score',
        'features': HEART_FEATURES,
    },
    18: {
        'path': 'models/kidney.pkl',
        'disease_name': 'Kidney Disease',
        'score_label': 'Kidney Disease Risk Score',
        'features': KIDNEY_FEATURES,
    },
    10: {
        'path': 'models/liver.pkl',
        'disease_name': 'Liver Disease',
        'score_label': 'Liver Disease Risk Score',
        'features': LIVER_FEATURES,
    },
}

TABULAR_MODELS = {}
DL_MODELS = {}


def load_tabular_model(model_path):
    if model_path not in TABULAR_MODELS:
        with open(model_path, 'rb') as model_file:
            TABULAR_MODELS[model_path] = pickle.load(model_file)
    return TABULAR_MODELS[model_path]


def load_dl_prediction_model(model_path):
    if model_path not in DL_MODELS:
        DL_MODELS[model_path] = load_model(model_path)
    return DL_MODELS[model_path]


def get_positive_probability(model, probabilities):
    if hasattr(model, 'classes_') and 1 in model.classes_:
        positive_index = list(model.classes_).index(1)
    else:
        positive_index = min(len(probabilities) - 1, 1)
    return float(probabilities[positive_index])


def get_default_step(value):
    magnitude = abs(float(value))
    if magnitude < 1:
        return 0.05
    if magnitude < 10:
        return 0.5
    if magnitude < 100:
        return 5.0
    return max(magnitude * 0.1, 10.0)


def get_candidate_values(current_value, feature_config):
    kind = feature_config.get('kind', 'continuous')
    current_value = float(current_value)

    if kind == 'binary':
        current_binary = 1.0 if current_value >= 0.5 else 0.0
        return [0.0 if current_binary == 1.0 else 1.0]

    if kind == 'choice':
        choices = [float(choice) for choice in feature_config.get('choices', [])]
        if not choices:
            return []

        ordered_choices = sorted(set(choices))
        current_choice = min(ordered_choices, key=lambda choice: abs(choice - current_value))
        candidates = []
        lower_choices = [choice for choice in ordered_choices if choice < current_choice]
        higher_choices = [choice for choice in ordered_choices if choice > current_choice]

        if lower_choices:
            candidates.append(lower_choices[-1])
        if higher_choices:
            candidates.append(higher_choices[0])
        if not candidates:
            candidates = [choice for choice in ordered_choices if not np.isclose(choice, current_choice)]
        return candidates

    step = float(feature_config.get('step', get_default_step(current_value)))
    min_value = feature_config.get('min')
    max_value = feature_config.get('max')

    lower_candidate = current_value - step
    if min_value is not None:
        lower_candidate = max(float(min_value), lower_candidate)
    elif current_value >= 0:
        lower_candidate = max(0.0, lower_candidate)

    higher_candidate = current_value + step
    if max_value is not None:
        higher_candidate = min(float(max_value), higher_candidate)

    candidates = []
    for candidate in [lower_candidate, higher_candidate]:
        if not np.isclose(candidate, current_value):
            candidates.append(float(candidate))

    return candidates


def format_feature_value(value):
    rounded_value = round(float(value), 2)
    if np.isclose(rounded_value, round(rounded_value)):
        return str(int(round(rounded_value)))
    return f"{rounded_value:.2f}".rstrip('0').rstrip('.')


def format_feature_list(feature_names):
    if not feature_names:
        return ''
    if len(feature_names) == 1:
        return feature_names[0]
    if len(feature_names) == 2:
        return f"{feature_names[0]} and {feature_names[1]}"
    return f"{', '.join(feature_names[:-1])}, and {feature_names[-1]}"


def build_global_feature_fallback(model, feature_configs, disease_name, excluded_features):
    if not hasattr(model, 'feature_importances_'):
        return []

    global_explanations = []
    ranked_features = sorted(
        zip(feature_configs, model.feature_importances_),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    for feature_config, importance in ranked_features:
        feature_label = feature_config.get('label', feature_config.get('key', 'Feature'))
        if feature_label in excluded_features or float(importance) <= 0:
            continue

        global_explanations.append({
            'feature': feature_label,
            'current_value': None,
            'direction': 'Current',
            'effect': 'shaped',
            'impact_points': round(float(importance) * 100, 2),
            'sentence': (
                f"{feature_label} is one of the model's strongest overall signals for "
                f"{disease_name.lower()}."
            ),
        })

    return global_explanations


def build_tabular_explanations(model, values, feature_configs, disease_name, pred, risk_score):
    if not hasattr(model, 'predict_proba'):
        return None, []

    base_values = np.asarray(values, dtype=np.float64).reshape(1, -1)
    base_probability = get_positive_probability(model, model.predict_proba(base_values)[0])
    explanations = []

    for index, feature_config in enumerate(feature_configs):
        current_value = float(values[index])
        candidates = get_candidate_values(current_value, feature_config)
        best_explanation = None

        for candidate_value in candidates:
            candidate_values = base_values.copy()
            candidate_values[0, index] = candidate_value
            candidate_probability = get_positive_probability(
                model,
                model.predict_proba(candidate_values)[0],
            )
            impact = base_probability - candidate_probability
            impact_points = abs(impact) * 100

            if impact_points == 0:
                continue

            if current_value > candidate_value:
                direction = 'Higher'
            elif current_value < candidate_value:
                direction = 'Lower'
            else:
                direction = 'Current'

            explanation = {
                'feature': feature_config.get('label', feature_config.get('key', f'Feature {index + 1}')),
                'current_value': format_feature_value(current_value),
                'direction': direction,
                'effect': 'increased' if impact > 0 else 'reduced',
                'impact_points': round(impact_points, 2),
            }
            explanation['sentence'] = (
                f"{explanation['direction']} {explanation['feature']} "
                f"(current value: {explanation['current_value']}) "
                f"{explanation['effect']} the score by about "
                f"{explanation['impact_points']:.2f} points."
            )

            if best_explanation is None or explanation['impact_points'] > best_explanation['impact_points']:
                best_explanation = explanation

        if best_explanation is not None:
            explanations.append(best_explanation)

    explanations.sort(key=lambda item: item['impact_points'], reverse=True)

    preferred_effect = 'increased' if pred == 1 else 'reduced'
    preferred = [item for item in explanations if item['effect'] == preferred_effect]
    fallback = [item for item in explanations if item['effect'] != preferred_effect]
    top_explanations = (preferred + fallback)[:3]

    if len(top_explanations) < 3:
        global_fallback = build_global_feature_fallback(
            model,
            feature_configs,
            disease_name,
            {item['feature'] for item in top_explanations},
        )
        top_explanations.extend(global_fallback[:3 - len(top_explanations)])

    if not top_explanations:
        return None, []

    summary = (
        f"The most influential inputs in this {disease_name.lower()} result were "
        f"{format_feature_list([item['feature'] for item in top_explanations])}."
    )

    if risk_score is not None:
        summary += (
            f" These are local model explanations based on how small changes in each input "
            f"would move the current risk score of {risk_score:.2f}%."
        )

    return summary, top_explanations


def predict(values):
    model_config = TABULAR_MODEL_CONFIG.get(len(values))
    if not model_config:
        raise ValueError('Unsupported prediction input length.')

    model_path = model_config['path']
    model = load_tabular_model(model_path)
    values = np.asarray(values, dtype=np.float64).reshape(1, -1)
    pred = int(model.predict(values)[0])
    risk_score = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(values)[0]
        risk_score = get_positive_probability(model, probabilities) * 100

    explanation_summary, feature_explanations = build_tabular_explanations(
        model,
        values[0],
        model_config.get('features', []),
        model_config['disease_name'],
        pred,
        risk_score,
    )

    return (
        pred,
        risk_score,
        model_config['score_label'],
        explanation_summary,
        feature_explanations,
    )


def predict_image(model_path, image_array):
    model = load_dl_prediction_model(model_path)
    probabilities = np.asarray(model.predict(image_array, verbose=0)[0]).squeeze()

    if np.ndim(probabilities) == 0:
        positive_probability = float(probabilities)
        pred = int(positive_probability >= 0.5)
    else:
        pred = int(np.argmax(probabilities))
        positive_probability = float(probabilities[min(len(probabilities) - 1, 1)])

    return pred, positive_probability * 100

@app.route("/")
def home():
    return render_template(
        'home.html',
        overview=HOME_PAGE_OVERVIEW,
        model_accuracies=MODEL_ACCURACIES,
        disease_details=DISEASE_DETAILS,
    )

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method != 'POST':
        return redirect(url_for('home'))

    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred, risk_score, score_label, explanation_summary, feature_explanations = predict(to_predict_list)
    except (TypeError, ValueError):
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template(
        'predict.html',
        pred=pred,
        risk_score=risk_score,
        score_label=score_label,
        explanation_summary=explanation_summary,
        feature_explanations=feature_explanations,
    )

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method != 'POST':
        return redirect(url_for('malariaPage'))

    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            raise ValueError('Missing image upload.')

        img = Image.open(request.files['image'])
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,3))
        img = img.astype(np.float64)
        pred, risk_score = predict_image("models/malaria.h5", img)
    except Exception:
        message = "Please upload an Image"
        return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred=pred, risk_score=risk_score)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method != 'POST':
        return redirect(url_for('pneumoniaPage'))

    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            raise ValueError('Missing image upload.')

        img = Image.open(request.files['image']).convert('L')
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,1))
        img = img / 255.0
        pred, risk_score = predict_image("models/pneumonia.h5", img)
    except Exception:
        message = "Please upload an Image"
        return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred=pred, risk_score=risk_score)

if __name__ == '__main__':
	app.run(debug = True)
