# data_analysis_app/views.py
import os
from django.shortcuts import render, HttpResponse
from data_analysis_project import settings
import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
from io import StringIO
import pickle

def data_tab(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")
 # Get the first ten rows of the dataset
    df_first_ten = df.head(10)
    df_last_ten = df.tail(10)

    # Get information about the columns (data types, non-null counts, null counts)
    columns_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })

    # Convert DataFrame to HTML for rendering in template
    table_html1 = df_first_ten.to_html(classes='table table-hover table-bordered')
    table_html2 = df_last_ten.to_html(classes='table table-hover table-bordered')

    # Include the columns information in the HTML template
    columns_info_html = f"{columns_info.to_html(classes='table table-hover table-bordered', index=False)}"

    return render(request, 'data_tab.html', {'table_html1': table_html1, 'table_html2':table_html2, 'columns_info_html': columns_info_html})



def profile_view(request):
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")

    # Create a profile report
    profile = ProfileReport(df, title="Pandas Profiling Report")
    
    print('Setting Dir Path')
    print(settings.TEMPLATE_DIR)
    templates_dir = settings.TEMPLATE_DIR
    report_path = os.path.join(templates_dir, 'report.html')

    # Save the report to the templates directory
    profile.to_file(report_path)
    # Pass the HTML file path to the template

    return render(request, 'report.html', {'report_path': report_path})


  
def descriptive_statistics_tab(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")
    
    # Perform descriptive statistics using pandas
    descriptive_stats = df.describe().to_html(classes='table table-bordered table-hover')

    return render(request, 'descriptive_statistics_tab.html', {'descriptive_stats': descriptive_stats})




def box_plot(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")

    # Default settings for the plot
    default_category = 'Pclass'
    default_value = 'Fare'

    # Get user-selected options (if any)
    selected_category = request.GET.get('category', default_category)
    selected_value = request.GET.get('value', default_value)

    # Validate selected features
    if selected_category not in df.columns or selected_value not in df.columns:
        error_message = "Invalid features selected for box plot."
        return render(request, 'error_page.html', {'error_message': error_message})

    # Create an interactive box plot using Plotly
    fig = px.box(df, x=selected_category, y=selected_value, title=f'Box Plot: {selected_value} by {selected_category}')

    # Convert the plot to HTML for rendering in the template
    plot_html = fig.to_html(full_html=False)

    # Pass parameters to the template for customization options
    box_cus_options = {
        'categories': df.columns.tolist(),
        'default_category': default_category,
        'default_value': default_value,
        'selected_category': selected_category,
        'selected_value': selected_value,
    }
    return {'plot_html': plot_html, 'box_cus_options': box_cus_options}
    #return render(request, 'box_plot.html', {'plot_html': plot_html, 'customization_options': customization_options})








def exploratory_data_analysis_tab(request):
    # Read your dataset
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")

    # Default settings for the plot
    default_feature = 'A.Murder '
    default_bins = 20

    # Get user-selected options (if any)
    selected_feature = request.GET.get('feature', default_feature)
    selected_bins = int(request.GET.get('bins', default_bins))

    # Create an interactive histogram using Plotly
    fig = px.histogram(df, x=selected_feature, nbins=selected_bins, title=f'{selected_feature} Distribution')

    # Convert the plot to HTML for rendering in the template
    plot_html = fig.to_html(full_html=False)

    # Pass parameters to the template for customization options
    customization_options = {
        'features': df.columns.tolist(),
        'default_feature': default_feature,
        'default_bins': default_bins,
        'selected_feature': selected_feature,
        'selected_bins': selected_bins,
    }

    # Default settings for the box plot
    default_category = 'State/UT'
    default_value = 'A.Murder '

    # Get user-selected options (if any)
    selected_category = request.GET.get('category', default_category)
    selected_value = request.GET.get('value', default_value)

    # Validate selected features
    if selected_category not in df.columns or selected_value not in df.columns:
        error_message = "Invalid features selected for box plot."
        return render(request, 'error_page.html', {'error_message': error_message})

    # Create an interactive box plot using Plotly
    fig_box = px.box(df, x=selected_category, y=selected_value, title=f'Box Plot: {selected_value} by {selected_category}')

    # Convert the box plot to HTML for rendering in the template
    box_plot_html = fig_box.to_html(full_html=False)

    # Pass parameters to the template for customization options
    box_cus_options = {
        'categories': df.columns.tolist(),
        'default_category': default_category,
        'default_value': default_value,
        'selected_category': selected_category,
        'selected_value': selected_value,
    }

    # Default settings for the scatter plot
    default_x_feature = 'A.Murder '
    default_y_feature = 'A. Culpable Homicide not amounting to Murder '

    # Get user-selected options (if any)
    selected_x_feature = request.GET.get('x_feature', default_x_feature)
    selected_y_feature = request.GET.get('y_feature', default_y_feature)

    # Create an interactive scatter plot using Plotly
    fig_scatter = px.scatter(df, x=selected_x_feature, y=selected_y_feature,
                             title=f'Scatter Plot: {selected_x_feature} vs. {selected_y_feature}')

    # Convert the scatter plot to HTML for rendering in the template
    plot_html_scatter = fig_scatter.to_html(full_html=False)

    # Pass parameters to the template for customization options
    customization_options_scatter = {
        'features': df.columns.tolist(),
        'default_x_feature': default_x_feature,
        'default_y_feature': default_y_feature,
        'selected_x_feature': selected_x_feature,
        'selected_y_feature': selected_y_feature,
    }

    # Default settings for the pie chart
    default_feature_pie = 'A.Murder '

    # Get user-selected options (if any)
    selected_feature_pie = request.GET.get('feature_pie', default_feature_pie)

    # Create an interactive pie chart using Plotly
    fig_pie = px.pie(df, names=selected_feature_pie, title=f'Pie Chart: {selected_feature_pie}')

    # Convert the pie chart to HTML for rendering in the template
    plot_html_pie = fig_pie.to_html(full_html=False)

    # Pass parameters to the template for customization options
    customization_options_pie = {
        'features_pie': df.columns.tolist(),
        'default_feature_pie': default_feature_pie,
        'selected_feature_pie': selected_feature_pie,
    }

    return render(request, 'exploratory_data_analysis_tab.html', {'plot_html': plot_html,
                                                                  'customization_options': customization_options,
                                                                  'box_plot_html': box_plot_html,
                                                                  'box_cus_options': box_cus_options,
                                                                  'plot_html_scatter': plot_html_scatter,
                                                                  'customization_options_scatter': customization_options_scatter,
                                                                  'plot_html_pie': plot_html_pie,
                                                                  'customization_options_pie': customization_options_pie,
                                                                  })












def export_to_csv(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")

    # Generate CSV file
    csv_file = df.to_csv(index=False)

    # Create HTTP response with CSV file
    response = HttpResponse(csv_file, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="titanic_data.csv"'
    
    return response


def export_to_excel(request):
    # Read Titanic dataset from CSV
    df = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")

    # Generate Excel file
    excel_file = BytesIO()
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_file.seek(0)

    # Create HTTP response with Excel file
    response = HttpResponse(excel_file.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="titanic_data.xlsx"'

    return response



import pandas as pd
from sklearn.model_selection import train_test_split  # Add this line
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def linear_regression_view(request):
    # Load and preprocess your data (replace this with your actual data loading and preprocessing code)
    data = pd.read_csv("C:\\Users\\HP\\Downloads\\DATA_PY.csv")
    feature1 = data["A. Culpable Homicide not amounting to Murder "].values.reshape(-1, 1)
    feature2 = data["A. Hurt"].values.reshape(-1, 1)
    target_variable = data["A.Murder "].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        pd.concat([pd.DataFrame(feature1), pd.DataFrame(feature2)], axis=1),
        target_variable, test_size=0.2, random_state=42
    )

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Given coefficients
    intercept = model.intercept_
    slope = model.coef_

    # Generate some example data for plotting
    X_plot = np.random.rand(100) * 10
    Y_plot = intercept + slope[0] * X_plot + np.random.randn(100) * 10

    # Calculate the predicted values for the regression line
    Y_pred_plot = intercept + slope[0] * X_plot

    # Plot the data points and the regression line
    plt.scatter(X_plot, Y_plot, label='Actual data')
    plt.plot(X_plot, Y_pred_plot, color='red', label='Regression line')
    plt.xlabel('Independent Variable (X)')
    plt.ylabel('Dependent Variable (Y)')
    plt.title('Simple Linear Regression')
    plt.legend()

    # Save the plot as an image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    # Prepare data for rendering in the template
    data_for_template = {
        'mse': mse,
        'r2': r2,
        'plot_image': img_data,
    }

    # Render the template with the data
    return render(request, 'linear_regression.html', data_for_template)




