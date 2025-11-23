"""
PAMAP2 Project - Required Libraries Installation
================================================

Run this in your terminal or Jupyter notebook:
pip install -r requirements.txt

Or install individually:
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly imbalanced-learn
"""

# requirements.txt content:
"""
# Data Processing
pandas==2.0.3
numpy==1.24.3

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1

# Machine Learning
scikit-learn==1.3.0
scipy==1.11.2
imbalanced-learn==0.11.0

# Additional utilities
tqdm==4.66.1
joblib==1.3.2
"""

# Verify installation script
def verify_installation():
    """Verify all required packages are installed"""
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'matplotlib': 'Basic plotting',
        'seaborn': 'Statistical visualization',
        'plotly': 'Interactive plots',
        'sklearn': 'Machine Learning',
        'scipy': 'Scientific computing',
        'imblearn': 'Handling imbalanced data'
    }
    
    print("🔍 Checking installed packages...\n")
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'imblearn':
                import imblearn
            else:
                __import__(package)
            print(f"✅ {package:15s} - {description}")
        except ImportError:
            print(f"❌ {package:15s} - NOT INSTALLED ({description})")
            all_installed = False
    
    if all_installed:
        print("\n✅ All packages installed successfully!")
    else:
        print("\n⚠️  Some packages are missing. Install them using:")
        print("pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy imbalanced-learn")
    
    return all_installed

if __name__ == "__main__":
    verify_installation()