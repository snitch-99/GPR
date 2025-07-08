import pandas as pd
from simplekml import Kml, Style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# Load your CSV
df = pd.read_csv('Data_2.csv')

# Normalize temperature values
min_temp = df['temperature'].min()
max_temp = df['temperature'].max()
norm = colors.Normalize(vmin=min_temp, vmax=max_temp)
colormap = cm.get_cmap('jet')  # blue to red

# Helper to convert RGBA to KML color (AABBGGRR)
def rgba_to_kml_color(rgba):
    r, g, b, a = [int(255 * x) for x in rgba]
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"

# Create KML object
kml = Kml()

# Add placemarks
for _, row in df.iterrows():
    temp = row['temperature']
    rgba = colormap(norm(temp))
    kml_color = rgba_to_kml_color(rgba)

    pnt = kml.newpoint(
        name=f"{temp:.1f} °C",
        coords=[(row['longitude'], row['latitude'])]
    )
    style = Style()
    style.iconstyle.color = kml_color
    style.iconstyle.scale = 1.2
    pnt.style = style

# Save to KML
kml.save("temperature_points.kml")
