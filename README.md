# RNN for AMOC Instability 🌊

This project explores the use of **Recurrent Neural Networks (RNN)**—specifically **Reservoir Computing (Echo State Networks)**—to detect and forecast tipping points in a simplified climate model representing the **Atlantic Meridional Overturning Circulation (AMOC)**.

We aim to anticipate **transitions to transient chaos and eventual system collapse** in the AMOC by training machine learning models on synthetic time series data generated from a coupled physical model.

---

## 🔬 About the Model

The system is based on a **5-dimensional box model** inspired by **Van Veen (2000)** and **Cessi (1994)**. It couples:
- Chaotic dynamics (x, y, z) modeled after Lorenz-style systems
- Ocean **temperature (T)** and **salinity (S)** dynamics
- External forcing (e.g., freshwater input) affecting system stability

---

## 🧠 Machine Learning Approach

The model is trained using:
- **Reservoir Computing (Echo State Networks)**
- Custom training loop in both **MATLAB** and **Python**
- Time series are generated across multiple bifurcation parameters
- Validation is performed via RMSE, prediction horizon, or error threshold

Key Features:
- Spectral radius scaling
- Multiple training trials (best-of strategy)
- Predictive generalization to unseen parameter regimes

---

## 📁 Structure

/matlab_code/ # Original dynamical model and training scripts (MATLAB) 

/python_code/ # Translated/rewritten training + RNN models (Python) 

/data/ # Placeholder for training/validation data 

/results/ # Placeholder for plots and predictions 

README.md # This file


---

## 📊 Example Output (coming soon!)

Plots of:
- Predicted vs. actual time series
- Relative prediction error over time
- RMSE across bifurcation parameters

---

## 🛠️ Tools Used

- MATLAB
- Python (NumPy, SciPy, Matplotlib)
- Custom RK4 ODE solvers
- Echo State Network implementation

---

## 🙌 You Can Help!

I'm currently cleaning up and documenting the code. If you're curious about:
- Machine learning for climate
- Reservoir computing
- Dynamical systems and tipping points

Feel free to **open an issue, ask a question, or contribute**!  
This repo is intended as a learning and collaboration space. Let's explore these tipping points together 🌍

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it.

---

## 📬 Contact

Made with curiosity and chaos in mind.  
Feel free to connect or collaborate!

