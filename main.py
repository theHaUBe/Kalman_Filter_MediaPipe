import cv2
import numpy as np
import mediapipe as mp

# https://gist.github.com/epiception/ac8195435976f6d2356869589b7157de
class Kalman_Filtering:
    def __init__(self, n_points):
        self.n_points = n_points

    def initialize(self):
        n_states = self.n_points * 4
        n_measures = self.n_points * 2
        self.kalman = cv2.KalmanFilter(n_states, n_measures)
        kalman = self.kalman
        kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        kalman.processNoiseCov = np.eye(n_states, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 1e-3

        kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
        dt = 1

        self.Measurement_array = []
        self.dt_array = []

        for i in range(0, n_states, 4):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)

        for i in range(0, n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        #print(self.dt_array)
        #print(self.Measurement_array)

        for i, j in zip(self.Measurement_array, self.dt_array):
            kalman.transitionMatrix[i, j] = dt

        for i in range(0, n_measures):
            kalman.measurementMatrix[i, self.Measurement_array[i]] = 1

        #print('TRANSITION Matrix:')
        #print(kalman.transitionMatrix)

        #print('MEASUREMENT Matrix:')
        #print(kalman.measurementMatrix)

    def predict(self, points):
        pred = []
        if points is not None:
            input_points = np.float32(np.ndarray.flatten(points))
            self.kalman.correct(input_points)
        tp = self.kalman.predict() 

        for i in self.Measurement_array:
            pred.append(tp[i])

        return np.array(pred).reshape(-1, 2)


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    #RYSOWANIE DANE
    mp_drawing = mp.solutions.drawing_utils
    drawing_color = (0, 255, 0)
    thickness = 2
    drawing_points = []

    kf = Kalman_Filtering(21)
    kf.initialize()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Nie udało się pobrać obrazu z kamery.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append([x, y])

                #RYSOWANIE
                thumb_tip = [hand_landmarks.landmark[4].x * frame.shape[1],
                             hand_landmarks.landmark[4].y * frame.shape[0]]
                index_tip = [hand_landmarks.landmark[8].x * frame.shape[1],
                             hand_landmarks.landmark[8].y * frame.shape[0]]
                distance = calculate_distance(thumb_tip, index_tip)
                if distance < 30:
                    drawing_points.append(tuple(map(int, index_tip)))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if landmarks:
            landmarks = np.array(landmarks, dtype=np.float32)
            predicted_landmarks = kf.predict(landmarks)

            for point in predicted_landmarks:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
        else:
            predicted_landmarks = kf.predict(np.zeros((21, 2), dtype=np.float32))

        if len(drawing_points) > 1:
            for i in range(1, len(drawing_points)):
                if drawing_points[i - 1] and drawing_points[i]:
                    cv2.line(frame, drawing_points[i - 1], drawing_points[i], drawing_color, thickness)

        cv2.imshow('Paint', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()