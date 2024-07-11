---
layout: post
title:  "How I Made the Map"
date:   2024-07-11 10:28:31 -0600
categories: jekyll update
---


To initialize the map, I used the module `folium` in python (found [here][folium]). Here is the script I used:
{% highlight ruby%}
import folium
from folium import IFrame

# Create a map centered around downtown calgary
m = folium.Map(location=[51.052944880632424, -114.0625643029516], zoom_start=11)

# Define the polygon locations
locationsSE = [
    [51.053,-114.0625],
    [50.85, -114.0625],
    [50.85, -113.86],
    [51.053, -113.86],
]

locationsNE = [
    [51.053,-114.0625],
    [51.192, -114.0625],
    [51.192, -113.86],
    [51.053, -113.86],
]

locationsNW = [
    [51.053,-114.0625],
    [51.192, -114.0625],
    [51.192, -114.3],
    [51.053, -114.3],
]

locationsSW = [
    [51.053,-114.0625],
    [50.85, -114.0625],
    [50.85, -114.3],
    [51.053, -114.3],
]

#Create the polygons
folium.Polygon(
    locations=locationsNW,
    color="blue",
    weight=0.1,
    fill_color="blue",
    fill_opacity=0.2,
    fill=True,
    tooltip="hello"
).add_to(m)

folium.Polygon(
    locations=locationsNE,
    color="orange",
    weight=0.1,
    fill_color="orange",
    fill_opacity=0.2,
    fill=True,
).add_to(m)

folium.Polygon(
    locations=locationsSE,
    color="red",
    weight=0.1,
    fill_color="red",
    fill_opacity=0.2,
    fill=True,
).add_to(m)

folium.Polygon(
    locations=locationsSW,
    color="purple",
    weight=0.1,
    fill_color="purple",
    fill_opacity=0.2,
    fill=True,
).add_to(m)

# Save the map
m.save("/Path/where/you/want/to/save/your/map.html")
{% endhighlight %}

Next, I opened the map in a text editor to edit the html. If you are following along with me, your code will likely look different from the code on my Github. This is expected, so don't worry.

 I first imported PapaParse so I could parse .csv and .txt files. To import it, I added the following to the lines that start with " <script src":
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>

I then created the info box that displays at the top right corner of the window.



[folium]:  https://pypi.org/project/folium/
[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
