from threading import Thread

import numpy as np
from flask import Flask, render_template_string, request, send_from_directory
from jinja2 import BaseLoader

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import (
    create_db, get_all_entries, get_timestamps,
    get_all_monitors, update_monitor_enabled, update_monitor_name
)
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable

app = Flask(__name__)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable

base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRecall</title>
  <!-- Bootstrap CSS -->
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
  <style>
    .slider-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    .slider {
      width: 80%;
    }
    .slider-value {
      margin-top: 10px;
      font-size: 1.2em;
    }
    .image-container {
      margin-top: 20px;
      text-align: center;
    }
    .image-container img {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <div class="container">
    <a class="navbar-brand" href="/">OpenRecall</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav mr-auto">
        <li class="nav-item">
          <a class="nav-link" href="/">Timeline</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/monitors">Monitors</a>
        </li>
      </ul>
      <form class="form-inline my-2 my-lg-0" action="/search" method="get">
        <input class="form-control mr-sm-2" type="search" name="q" placeholder="Search" aria-label="Search">
        <button class="btn btn-outline-secondary my-2 my-sm-0" type="submit">
          <i class="bi bi-search"></i>
        </button>
      </form>
    </div>
  </div>
</nav>
{% block content %}

{% endblock %}

  <!-- Bootstrap and jQuery JS -->
  <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>

</body>
</html>
"""


class StringLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == "base_template":
            return base_template, None, lambda: True
        return None, None, None


app.jinja_env.loader = StringLoader()


@app.route("/")
def timeline():
    # Get all entries and group by timestamp
    entries = get_all_entries()

    # Group entries by timestamp (descending order)
    from collections import defaultdict
    timestamp_groups = defaultdict(list)
    for entry in entries:
        timestamp_groups[entry.timestamp].append(entry)

    # Get unique timestamps in descending order
    timestamps = sorted(timestamp_groups.keys(), reverse=True)

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
{% if timestamps|length > 0 %}
  <div class="container">
    <div class="slider-container">
      <input type="range" class="slider custom-range" id="discreteSlider" min="0" max="{{timestamps|length - 1}}" step="1" value="{{timestamps|length - 1}}">
      <div class="slider-value" id="sliderValue">{{timestamps[0] | timestamp_to_human_readable }}</div>
    </div>
    <div class="image-container" id="imageContainer">
      <!-- Images will be inserted here by JavaScript -->
    </div>
  </div>
  <script>
    const timestampGroups = {{ timestamp_groups|tojson }};
    const timestamps = {{ timestamps|tojson }};
    const slider = document.getElementById('discreteSlider');
    const sliderValue = document.getElementById('sliderValue');
    const imageContainer = document.getElementById('imageContainer');

    function updateImages(timestamp) {
      const entries = timestampGroups[timestamp];
      imageContainer.innerHTML = '';

      if (entries && entries.length > 0) {
        entries.forEach(entry => {
          const wrapper = document.createElement('div');
          wrapper.style.marginBottom = '20px';

          const label = document.createElement('div');
          label.className = 'text-muted mb-2';
          label.textContent = entry.monitor_name || 'Monitor ' + entry.monitor_id;

          const img = document.createElement('img');
          img.src = `/static/${timestamp}_${entry.monitor_id}.webp`;
          img.alt = `Screenshot from ${entry.monitor_name}`;
          img.style.maxWidth = '100%';
          img.style.height = 'auto';
          img.style.border = '1px solid #ddd';
          img.style.borderRadius = '4px';
          img.style.padding = '5px';

          wrapper.appendChild(label);
          wrapper.appendChild(img);
          imageContainer.appendChild(wrapper);
        });
      }
    }

    slider.addEventListener('input', function() {
      const reversedIndex = timestamps.length - 1 - slider.value;
      const timestamp = timestamps[reversedIndex];
      sliderValue.textContent = new Date(timestamp * 1000).toLocaleString();
      updateImages(timestamp);
    });

    // Initialize the slider with the most recent timestamp
    slider.value = timestamps.length - 1;
    sliderValue.textContent = new Date(timestamps[0] * 1000).toLocaleString();
    updateImages(timestamps[0]);
  </script>
{% else %}
  <div class="container">
      <div class="alert alert-info" role="alert">
          Nothing recorded yet, wait a few seconds.
      </div>
  </div>
{% endif %}
{% endblock %}
""",
        timestamps=timestamps,
        timestamp_groups={str(k): v for k, v in timestamp_groups.items()},
    )


@app.route("/search")
def search():
    q = request.args.get("q")
    entries = get_all_entries()
    embeddings = [np.frombuffer(entry.embedding, dtype=np.float32) for entry in entries]
    query_embedding = get_embedding(q)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)[::-1]
    sorted_entries = [entries[i] for i in indices]

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
    <div class="container">
        <h3 class="mt-3 mb-4">Search Results for "{{ query }}"</h3>
        <div class="row">
            {% for entry in entries %}
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}">
                            <img src="/static/{{ entry.timestamp }}_{{ entry.monitor_id }}.webp" alt="Image" class="card-img-top">
                        </a>
                        <div class="card-body">
                            <small class="text-muted">
                                {{ entry.monitor_name }}<br>
                                {{ entry.timestamp | timestamp_to_human_readable }}
                            </small>
                        </div>
                    </div>
                </div>
                <div class="modal fade" id="modal-{{ loop.index0 }}" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-xl" role="document" style="max-width: none; width: 100vw; height: 100vh; padding: 20px;">
                        <div class="modal-content" style="height: calc(100vh - 40px); width: calc(100vw - 40px); padding: 0;">
                            <div class="modal-header">
                                <h5 class="modal-title">{{ entry.monitor_name }} - {{ entry.timestamp | timestamp_to_human_readable }}</h5>
                                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                    <span aria-hidden="true">&times;</span>
                                </button>
                            </div>
                            <div class="modal-body" style="padding: 0;">
                                <img src="/static/{{ entry.timestamp }}_{{ entry.monitor_id }}.webp" alt="Image" style="width: 100%; height: 100%; object-fit: contain; margin: 0 auto;">
                            </div>
                        </div>
                    </div>
                </div>
            {% endfor %}
        </div>
    </div>
{% endblock %}
""",
        entries=sorted_entries,
        query=q,
    )


@app.route("/monitors")
def monitors():
    monitors = get_all_monitors()
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
    <div class="container mt-4">
        <h2>Monitor Configuration</h2>
        <p class="text-muted">Select which monitors to capture screenshots from. You can also customize monitor names.</p>

        {% if monitors|length > 0 %}
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Monitor</th>
                        <th>Resolution</th>
                        <th>Name</th>
                        <th>Enabled</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for monitor in monitors %}
                    <tr id="monitor-row-{{ monitor.monitor_index }}">
                        <td>Monitor {{ monitor.monitor_index }}</td>
                        <td>{{ monitor.width }} x {{ monitor.height }}</td>
                        <td>
                            <span id="name-display-{{ monitor.monitor_index }}">{{ monitor.name }}</span>
                            <input type="text" class="form-control d-none" id="name-input-{{ monitor.monitor_index }}" value="{{ monitor.name }}">
                        </td>
                        <td>
                            <div class="custom-control custom-switch">
                                <input type="checkbox" class="custom-control-input" id="enabled-{{ monitor.monitor_index }}"
                                    {% if monitor.enabled %}checked{% endif %}
                                    onchange="toggleMonitor({{ monitor.monitor_index }}, this.checked)">
                                <label class="custom-control-label" for="enabled-{{ monitor.monitor_index }}"></label>
                            </div>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-primary" id="edit-btn-{{ monitor.monitor_index }}"
                                onclick="editMonitorName({{ monitor.monitor_index }})">
                                <i class="bi bi-pencil"></i> Edit
                            </button>
                            <button class="btn btn-sm btn-success d-none" id="save-btn-{{ monitor.monitor_index }}"
                                onclick="saveMonitorName({{ monitor.monitor_index }})">
                                <i class="bi bi-check"></i> Save
                            </button>
                            <button class="btn btn-sm btn-secondary d-none" id="cancel-btn-{{ monitor.monitor_index }}"
                                onclick="cancelEditMonitorName({{ monitor.monitor_index }}, '{{ monitor.name }}')">
                                Cancel
                            </button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <div class="alert alert-info" role="alert">
            No monitors detected yet. Monitors will be detected when the screenshot service starts.
        </div>
        {% endif %}
    </div>

    <script>
        function toggleMonitor(monitorIndex, enabled) {
            fetch('/api/monitors/' + monitorIndex + '/enabled', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enabled: enabled })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Monitor ' + monitorIndex + ' enabled status updated to ' + enabled);
                } else {
                    alert('Failed to update monitor status');
                    // Revert checkbox
                    document.getElementById('enabled-' + monitorIndex).checked = !enabled;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error updating monitor status');
                // Revert checkbox
                document.getElementById('enabled-' + monitorIndex).checked = !enabled;
            });
        }

        function editMonitorName(monitorIndex) {
            document.getElementById('name-display-' + monitorIndex).classList.add('d-none');
            document.getElementById('name-input-' + monitorIndex).classList.remove('d-none');
            document.getElementById('edit-btn-' + monitorIndex).classList.add('d-none');
            document.getElementById('save-btn-' + monitorIndex).classList.remove('d-none');
            document.getElementById('cancel-btn-' + monitorIndex).classList.remove('d-none');
        }

        function cancelEditMonitorName(monitorIndex, originalName) {
            document.getElementById('name-input-' + monitorIndex).value = originalName;
            document.getElementById('name-display-' + monitorIndex).classList.remove('d-none');
            document.getElementById('name-input-' + monitorIndex).classList.add('d-none');
            document.getElementById('edit-btn-' + monitorIndex).classList.remove('d-none');
            document.getElementById('save-btn-' + monitorIndex).classList.add('d-none');
            document.getElementById('cancel-btn-' + monitorIndex).classList.add('d-none');
        }

        function saveMonitorName(monitorIndex) {
            const newName = document.getElementById('name-input-' + monitorIndex).value;

            fetch('/api/monitors/' + monitorIndex + '/name', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: newName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('name-display-' + monitorIndex).textContent = newName;
                    document.getElementById('name-display-' + monitorIndex).classList.remove('d-none');
                    document.getElementById('name-input-' + monitorIndex).classList.add('d-none');
                    document.getElementById('edit-btn-' + monitorIndex).classList.remove('d-none');
                    document.getElementById('save-btn-' + monitorIndex).classList.add('d-none');
                    document.getElementById('cancel-btn-' + monitorIndex).classList.add('d-none');
                    // Update cancel button to use new name
                    document.getElementById('cancel-btn-' + monitorIndex).setAttribute('onclick',
                        "cancelEditMonitorName(" + monitorIndex + ", '" + newName + "')");
                } else {
                    alert('Failed to update monitor name');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error updating monitor name');
            });
        }
    </script>
{% endblock %}
""",
        monitors=monitors,
    )


@app.route("/api/monitors/<int:monitor_index>/enabled", methods=["POST"])
def update_monitor_enabled_api(monitor_index):
    data = request.get_json()
    enabled = data.get("enabled", False)
    success = update_monitor_enabled(monitor_index, enabled)
    return {"success": success}


@app.route("/api/monitors/<int:monitor_index>/name", methods=["POST"])
def update_monitor_name_api(monitor_index):
    data = request.get_json()
    name = data.get("name", "")
    if not name:
        return {"success": False, "error": "Name cannot be empty"}
    success = update_monitor_name(monitor_index, name)
    return {"success": success}


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
