"""
CELL 25: INTERACTIVE WEB INTERFACE WITH GRADIO
Beautiful frontend for face recognition system
"""

import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

# Global variables for loaded models
loaded_models = {
    'yale': {
        'HOG+SVM': hog_svm_yale,
        'PCA+RF': pca_rf_yale,
        'CNN': cnn_yale,
        'VGG16': vgg_yale
    },
    'lfw': {
        'HOG+SVM': hog_svm_lfw,
        'PCA+RF': pca_rf_lfw,
        'CNN': cnn_lfw,
        'VGG16': vgg_lfw
    }
}

loaded_labels = {
    'yale': yale_names,
    'lfw': lfw_names
}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_face(image, dataset_choice, model_choice):
    """
    Predict face identity from uploaded image

    Args:
        image: PIL Image or numpy array
        dataset_choice: 'Yale' or 'LFW'
        model_choice: 'HOG+SVM', 'PCA+RF', 'CNN', or 'VGG16'

    Returns:
        Prediction results and visualization
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess image
        preprocessor = FacePreprocessor(target_size=(128, 128))

        # Convert PIL to path-like (save temporarily)
        temp_path = '/tmp/temp_face.jpg'
        image.save(temp_path)

        # Preprocess
        img_processed = preprocessor.preprocess_image(temp_path)

        if img_processed is None:
            return "Error: Could not process image", None, None

        # Get model and labels
        dataset_key = dataset_choice.lower()
        model = loaded_models[dataset_key][model_choice]
        label_names = loaded_labels[dataset_key]

        # Prepare input
        img_input = np.expand_dims(img_processed, axis=0)

        # Predict
        prediction = model.predict(img_input)[0]
        predicted_label = label_names[prediction]

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(img_input)[0]

            # Get top 5 predictions
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            top_5_labels = [label_names[i] for i in top_5_idx]
            top_5_probs = probabilities[top_5_idx]

            # Create probability bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]
            bars = ax.barh(range(5), top_5_probs, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(5))
            ax.set_yticklabels(top_5_labels, fontsize=12)
            ax.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
            ax.set_title(f'Top 5 Predictions - {model_choice} ({dataset_choice})',
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlim([0, 1])
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, top_5_probs)):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob*100:.2f}%',
                       ha='left', va='center', fontsize=11, fontweight='bold')

            plt.tight_layout()

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            prob_chart = Image.open(buf)
            plt.close()

            confidence = probabilities[prediction]

        except:
            prob_chart = None
            confidence = 1.0

        # Create result text
        result_text = f"""
        🎯 **Prediction Result**

        **Predicted Identity:** {predicted_label}
        **Confidence:** {confidence*100:.2f}%
        **Model Used:** {model_choice}
        **Dataset:** {dataset_choice}
        **Total Classes:** {len(label_names)}
        """

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=13, fontweight='bold')
        axes[0].axis('off')

        # Preprocessed image
        axes[1].imshow(img_processed)
        axes[1].set_title(f'Preprocessed (128x128)\nPredicted: {predicted_label}',
                         fontsize=13, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        comparison_img = Image.open(buf)
        plt.close()

        return result_text, comparison_img, prob_chart

    except Exception as e:
        return f"Error: {str(e)}", None, None

# ============================================================================
# DATASET EXPLORATION FUNCTION
# ============================================================================

def show_dataset_samples(dataset_choice, num_samples=10):
    """Show random samples from selected dataset"""
    try:
        dataset_key = dataset_choice.lower()

        if dataset_key == 'yale':
            dataset_path = YALE_DIR
        else:
            dataset_path = LFW_FILTERED_DIR

        subjects = os.listdir(dataset_path)
        selected_subjects = np.random.choice(subjects, min(num_samples, len(subjects)), replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        fig.suptitle(f'{dataset_choice} Dataset - Random Samples',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, subject in enumerate(selected_subjects):
            subject_path = os.path.join(dataset_path, subject)
            if os.path.isdir(subject_path):
                images = [f for f in os.listdir(subject_path) if f.endswith('.jpg')]
                if images:
                    img_path = os.path.join(subject_path, images[0])
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].set_title(subject[:20], fontsize=10, fontweight='bold')
                    axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(selected_subjects), 10):
            axes[i].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        samples_img = Image.open(buf)
        plt.close()

        return samples_img

    except Exception as e:
        return None

# ============================================================================
# PERFORMANCE METRICS FUNCTION
# ============================================================================

def show_performance_metrics(dataset_choice, model_choice):
    """Display performance metrics for selected model and dataset"""
    try:
        # Filter results
        dataset_env = 'Controlled' if dataset_choice == 'Yale' else 'Uncontrolled'

        filtered_results = results_df[
            (results_df['Environment'] == dataset_env) &
            (results_df['Model_Type'] == model_choice)
        ]

        if filtered_results.empty:
            return "No results found for this combination"

        row = filtered_results.iloc[0]

        metrics_text = f"""
        📊 **Performance Metrics**

        **Model:** {model_choice}
        **Dataset:** {dataset_choice} ({dataset_env})

        **Metrics:**
        - Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)
        - Precision: {row['Precision']:.4f}
        - Recall: {row['Recall']:.4f}
        - F1-Score: {row['F1-Score']:.4f}
        """

        # Create metrics visualization
        fig, ax = plt.subplots(figsize=(8, 6))

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_choice} Performance - {dataset_choice} Dataset',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        metrics_img = Image.open(buf)
        plt.close()

        return metrics_text, metrics_img

    except Exception as e:
        return f"Error: {str(e)}", None

# ============================================================================
# MODEL COMPARISON FUNCTION
# ============================================================================

def compare_all_models():
    """Show comparison of all models"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Models Performance Comparison',
                     fontsize=18, fontweight='bold', y=0.995)

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            pivot_df = results_df.pivot_table(
                values=metric,
                index='Model_Type',
                columns='Environment'
            )

            x = np.arange(len(pivot_df.index))
            width = 0.35

            bars1 = ax.bar(x - width/2, pivot_df['Controlled'], width,
                          label='Controlled', color='#3498db', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, pivot_df['Uncontrolled'], width,
                          label='Uncontrolled', color='#e74c3c', alpha=0.8, edgecolor='black')

            ax.set_title(f'{metric}', fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=10)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)

            # Add labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        comparison_img = Image.open(buf)
        plt.close()

        return comparison_img

    except Exception as e:
        return None

# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

print("🎨 Building Gradio Interface...")

with gr.Blocks(theme=gr.themes.Soft(), title="Face Recognition System") as demo:

    gr.Markdown("""
    # 🎭 Face Recognition: Controlled vs Uncontrolled Environments
    ### Interactive Demo - Compare multiple ML/DL models on Yale and LFW datasets
    """)

    with gr.Tabs():

        # ========================================================================
        # TAB 1: FACE RECOGNITION
        # ========================================================================
        with gr.TabItem("🔍 Face Recognition"):
            gr.Markdown("### Upload an image to recognize the face")

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Face Image")

                    with gr.Row():
                        dataset_select = gr.Radio(
                            choices=["Yale", "LFW"],
                            value="Yale",
                            label="Select Dataset",
                            info="Choose which dataset's models to use"
                        )

                        model_select = gr.Radio(
                            choices=["HOG+SVM", "PCA+RF", "CNN", "VGG16"],
                            value="VGG16",
                            label="Select Model",
                            info="Choose recognition model"
                        )

                    predict_btn = gr.Button("🎯 Recognize Face", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_text = gr.Markdown(label="Prediction Result")
                    comparison_image = gr.Image(label="Image Comparison")

            with gr.Row():
                probability_chart = gr.Image(label="Confidence Scores (Top 5)")

            predict_btn.click(
                fn=predict_face,
                inputs=[input_image, dataset_select, model_select],
                outputs=[result_text, comparison_image, probability_chart]
            )

            gr.Markdown("""
            **Instructions:**
            1. Upload a face image
            2. Select dataset (Yale for controlled, LFW for uncontrolled)
            3. Choose a model (VGG16 recommended for best accuracy)
            4. Click 'Recognize Face' to see prediction
            """)

        # ========================================================================
        # TAB 2: DATASET EXPLORER
        # ========================================================================
        with gr.TabItem("📊 Dataset Explorer"):
            gr.Markdown("### Explore the datasets used in this project")

            with gr.Row():
                dataset_explore = gr.Radio(
                    choices=["Yale", "LFW"],
                    value="Yale",
                    label="Select Dataset to Explore"
                )
                explore_btn = gr.Button("🔄 Show Random Samples", variant="primary")

            dataset_samples = gr.Image(label="Dataset Samples")

            explore_btn.click(
                fn=show_dataset_samples,
                inputs=[dataset_explore],
                outputs=[dataset_samples]
            )

            gr.Markdown("""
            **Dataset Information:**

            **Yale Face Database (Controlled):**
            - Studio lighting, frontal poses
            - Uniform backgrounds
            - Minimal variations

            **LFW Database (Uncontrolled):**
            - Natural lighting conditions
            - Various poses and angles
            - Complex backgrounds
            - Real-world variations
            """)

        # ========================================================================
        # TAB 3: MODEL PERFORMANCE
        # ========================================================================
        with gr.TabItem("📈 Model Performance"):
            gr.Markdown("### View detailed performance metrics for each model")

            with gr.Row():
                perf_dataset = gr.Radio(
                    choices=["Yale", "LFW"],
                    value="Yale",
                    label="Select Dataset"
                )
                perf_model = gr.Radio(
                    choices=["HOG+SVM", "PCA+RF", "CNN", "VGG16"],
                    value="VGG16",
                    label="Select Model"
                )

            show_metrics_btn = gr.Button("📊 Show Metrics", variant="primary")

            with gr.Row():
                metrics_text = gr.Markdown(label="Metrics Details")
                metrics_chart = gr.Image(label="Performance Visualization")

            show_metrics_btn.click(
                fn=show_performance_metrics,
                inputs=[perf_dataset, perf_model],
                outputs=[metrics_text, metrics_chart]
            )

        # ========================================================================
        # TAB 4: MODEL COMPARISON
        # ========================================================================
        with gr.TabItem("⚖️ Compare All Models"):
            gr.Markdown("### Compare performance of all models across both datasets")

            compare_btn = gr.Button("📊 Generate Comparison", variant="primary", size="lg")
            comparison_result = gr.Image(label="All Models Comparison")

            compare_btn.click(
                fn=compare_all_models,
                inputs=[],
                outputs=[comparison_result]
            )

            # Show results table
            gr.Markdown("### Detailed Results Table")
            results_table = gr.Dataframe(
                value=results_df[['Model', 'Environment', 'Accuracy', 'Precision', 'Recall', 'F1-Score']],
                label="Complete Results"
            )

        # ========================================================================
        # TAB 5: ABOUT
        # ========================================================================
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(f"""
            # About This Project

            ## 🎯 Objective
            Compare face recognition performance across **controlled** and **uncontrolled**
            environments using multiple Machine Learning and Deep Learning approaches.

            ## 📊 Datasets

            **1. Yale Face Database (Controlled)**
            - Subjects: {len(yale_names)}
            - Total Images: {len(X_yale)}
            - Environment: Studio conditions

            **2. Labeled Faces in the Wild - LFW (Uncontrolled)**
            - Subjects: {len(lfw_names)}
            - Total Images: {len(X_lfw)}
            - Environment: Real-world conditions

            ## 🤖 Models Implemented

            1. **HOG + SVM**: Traditional computer vision approach
            2. **PCA + Random Forest**: Dimensionality reduction + Ensemble learning
            3. **Custom CNN**: Deep learning with custom architecture
            4. **VGG16 Transfer Learning**: Pre-trained deep neural network

            ## 📈 Key Findings

            - **Best Controlled**: {best_controlled['Model'].values[0]} ({best_controlled['Accuracy'].values[0]:.2%})
            - **Best Uncontrolled**: {best_uncontrolled['Model'].values[0]} ({best_uncontrolled['Accuracy'].values[0]:.2%})
            - **Most Robust**: {most_robust['Model']} (Drop: {most_robust['Drop_Percentage']:.2f}%)

            ## 🛠️ Technologies Used

            - **Frontend**: Gradio
            - **ML/DL**: Scikit-learn, TensorFlow/Keras
            - **Computer Vision**: OpenCV, scikit-image
            - **Visualization**: Matplotlib, Seaborn

            ## 📝 Citation

            If you use this work, please cite:
            - Yale Face Database
            - Labeled Faces in the Wild (LFW) dataset

            ---

            **Created with ❤️ for ML/DL Face Recognition Research**
            """)

print("✅ Gradio interface built successfully!")

# Launch the interface
print("\n" + "="*80)
print("🚀 LAUNCHING WEB INTERFACE...")
print("="*80)

demo.launch(
    share=True,  # Create public link
    debug=True,
    show_error=True
)
