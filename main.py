import argparse
from VehicleInference.inference import VehicleInference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detector")
    parser.add_argument("--source", type=str, help="Path to the images")

    args = parser.parse_args()

    vehicle_inference = VehicleInference()
    vehicle_inference.initialize_model(path_model="/home/willian/PycharmProjects/vehicle_inference/best.pt")
    vehicle_inference.inference(source=args.source)

