How to run model training pipeline?

    1. Create venv
    2. Install required packages : "pip install -r requiements.txt"
    3. Ensure raw data path in config/config.yaml
    4. Run 'python main.py', and wait for the model training and evaluation results in terminal.
        - The process generates artifacts and objects in artifacts folder / Recommended to delete artifacts folder for a fresh run/checks

How to run predictions?

    1. Run 'python app.py'
    2. Navigate/listen to http://127.0.0.1:5000/ in browser
    3. Enter your parameters in the form and checkout the prediction !

