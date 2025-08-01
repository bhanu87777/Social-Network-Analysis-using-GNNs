from train import train_model
from train import train
from evaluate import evaluate_model
from visualization import visualize_graph, visualize_embeddings
from logistic_regression import run_logistic_regression
from graphsage import GraphSAGE
from parameter_comparison import run_experiment
from new_sampling import run_new_sampling_experiment

def main():

    print("Training the model...")
    train_model()

    print("\nEvaluating the model...")
    evaluate_model()

    print("\nVisualizing the results...")
    visualize_graph()
    visualize_embeddings()

    print("Training Logistic Regression...")
    lr_accuracy = run_logistic_regression()
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%\n")

    print("Running GraphSAGE...")
    graphsage_accuracy = train()
    print(f"GraphSAGE Accuracy: {graphsage_accuracy:.2f}%\n")
    
    print("\nRunning parameter comparison experiment...")
    run_experiment()

    print(f"Accuracy for Fixed Number: {53.87}%")
    run_new_sampling_experiment()

if __name__ == "__main__":
    main()