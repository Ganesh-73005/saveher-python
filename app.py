from flask import Flask, request, jsonify
import joblib
import numpy as np
import openrouteservice as ors
from openrouteservice.directions import directions
import polyline
app = Flask(__name__)

# Load the risk model
predictor = joblib.load("risk_model.pkl")
gmm = predictor['gmm']
scaler = predictor['scaler']

# Initialize OpenRouteService client
ORS_API_KEY = "5b3ce3597851110001cf6248e4fb722f36894ddfb2325debdbd77c4e"
ors_client = ors.Client(key=ORS_API_KEY)

def calculate_point_risk(lat, lng):
    """Calculate risk score for a specific point using the trained GMM."""
    try:
        point = scaler.transform([[lat, lng]])[0]
        gmm_density = np.exp(gmm.score_samples([[lat, lng]]))
        return gmm_density[0]
    except Exception as e:
        return None

def get_safe_route(src, dest):
    """Get the safest route based on risk scores."""
    try:
        # Ensure coordinates are in the correct order [longitude, latitude]
        coords = [
            [src['longitude'], src['latitude']],  # [longitude, latitude]
            [dest['longitude'], dest['latitude']]
        ]
        
        # Request multiple routes with different preferences
        route_preferences = ['recommended', 'shortest', 'fastest']
        routes = []
        
        for preference in route_preferences:
            route = ors_client.directions(
                coordinates=coords,
                profile='foot-walking',
                format='geojson',
                preference=preference
            )
            routes.append(route)
        
        safest_route = None
        lowest_risk = float('inf')
        
        # Iterate through each route and calculate the total risk
        for route in routes:
            total_risk = 0
            path = []
            
            # Check each point in the route
            for coord in route['features'][0]['geometry']['coordinates']:
                lng, lat = coord
                risk_score = calculate_point_risk(lat, lng)
                total_risk += risk_score if risk_score is not None else 0
                path.append((lat, lng))  # Append coordinates (latitude, longitude)
            
            # If this route has the lowest risk, set it as the safest route
            if total_risk < lowest_risk:
                lowest_risk = total_risk
                safest_route = path
        
        # Convert the safest route into a polyline
        if safest_route:
            # Use the polyline library to encode the route
            encoded_polyline = polyline.encode(safest_route)
            return encoded_polyline
        else:
            raise ValueError("No safe route found")
    
    except Exception as e:
        print(f"Error in get_safe_route: {e}")
        return None

@app.route("/get_safe_route", methods=["POST"])
def api_get_safe_route():
    data = request.json
    src = data.get("src")
    dest = data.get("dest")
    print(src,dest)
    if not src or not dest:
        return jsonify({"error": "Missing source or destination"}), 400

    try:
        # Ensure src and dest are dictionaries with 'latitude' and 'longitude' keys
        if not all(key in src for key in ['latitude', 'longitude']) or \
           not all(key in dest for key in ['latitude', 'longitude']):
            return jsonify({"error": "Invalid format for source or destination"}), 400

        safe_polyline = get_safe_route(src, dest)
        if safe_polyline:
            return jsonify({"safest_polyline": safe_polyline})
        else:
            return jsonify({"error": "Unable to fetch safe route"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(debug=True)


