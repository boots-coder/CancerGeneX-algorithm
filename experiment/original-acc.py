import plotly.graph_objects as go

# Data
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy = [
    0.8175287356321839,
    0.8577586206896551,
    0.8433908045977011,
    0.8721264367816092,
    0.8879310344827587,
    0.8663793103448276,
    0.8951149425287356,
    0.8793103448275862,
    0.8864942528735632,
    0.8419540229885057
]

# Bar chart with holiday theme
fig = go.Figure(
    data=[
        go.Bar(
            x=layers,
            y=accuracy,
            marker=dict(
                color=['#FF0000', '#00FF00', '#FF4500', '#FFD700', '#228B22',
                       '#FF69B4', '#8A2BE2', '#5F9EA0', '#DC143C', '#008B8B'],
                line=dict(color='black', width=1)
            ),
            text=[f"{acc:.2%}" for acc in accuracy],  # Display accuracy as percentages
            textposition='outside',
            name='Accuracy'
        )
    ]
)

# Layout customization
fig.update_layout(
    title="🎄 Holiday-Themed Accuracy per Layer 🎁",
    title_font=dict(size=24, color='darkgreen'),
    xaxis=dict(title="Layer", title_font=dict(size=18)),
    yaxis=dict(title="Accuracy", title_font=dict(size=18)),
    plot_bgcolor='#FFFACD',  # Background color
    paper_bgcolor='#FFFACD',  # Outer background
    font=dict(family="Arial", size=14),
    showlegend=False
)

# Save as HTML file
fig.write_html("holiday_chart.html")
print("Chart saved as 'holiday_chart.html'. You can upload this file to share it online.")