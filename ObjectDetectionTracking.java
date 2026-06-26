import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ObjectDetectionTracking {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static class Detection {
        Rect2d box;
        float score;
        int classId;

        Detection(Rect2d box, float score, int classId) {
            this.box = box;
            this.score = score;
            this.classId = classId;
        }
    }

    private static class Track {
        Rect2d box;
        int id;
        int classId;
        float score;

        Track(Rect2d box, int id, int classId, float score) {
            this.box = box;
            this.id = id;
            this.classId = classId;
            this.score = score;
        }
    }

    private static class KalmanBoxTracker {
        private final double[][] F;
        private final double[][] H;
        private final double[][] P;
        private final double[][] Q;
        private final double[][] R;
        private double[] x;
        int id;
        int classId;
        float score;
        int timeSinceUpdate = 0;
        int hits = 0;
        int hitStreak = 0;
        int age = 0;

        KalmanBoxTracker(Rect2d bbox, int id, int classId, float score) {
            this.id = id;
            this.classId = classId;
            this.score = score;
            x = convertToCenter(bbox);
            F = new double[7][7];
            H = new double[4][7];
            P = new double[7][7];
            Q = new double[7][7];
            R = new double[4][4];
            initMatrices();
            for (int i = 0; i < 7; i++) {
                P[i][i] = (i < 4) ? 10 : 1000;
            }
            hits = 1;
            hitStreak = 1;
        }

        private void initMatrices() {
            for (int i = 0; i < 7; i++) {
                F[i][i] = 1.0;
            }
            F[0][4] = 1.0;
            F[1][5] = 1.0;
            F[2][6] = 1.0;
            for (int i = 0; i < 4; i++) {
                H[i][i] = 1.0;
            }
            for (int i = 0; i < 4; i++) {
                R[i][i] = 1.0;
            }
            Q[4][4] = 0.01;
            Q[5][5] = 0.01;
            Q[6][6] = 0.01;
        }

        void predict() {
            x = add(matMul(F, x), new double[7]);
            P = add(matMul(F, matMul(P, transpose(F))), Q);
            age += 1;
            if (timeSinceUpdate > 0) {
                hitStreak = 0;
            }
            timeSinceUpdate += 1;
        }

        void update(Rect2d bbox, int classId, float score) {
            double[] z = convertToCenter(bbox);
            double[] y = subtract(z, matMul(H, x));
            double[][] S = add(matMul(H, matMul(P, transpose(H))), R);
            double[][] K = matMul(matMul(P, transpose(H)), inverse(S));
            x = add(x, matMul(K, y));
            double[][] I = identityMatrix(7);
            P = matMul(subtract(I, matMul(K, H)), P);
            timeSinceUpdate = 0;
            hits += 1;
            hitStreak += 1;
            this.classId = classId;
            this.score = score;
        }

        Rect2d getState() {
            double[] center = x;
            double cx = center[0];
            double cy = center[1];
            double s = center[2];
            double r = Math.max(center[3], 1e-6);
            double w = Math.sqrt(Math.max(s * r, 0.0));
            double h = Math.sqrt(Math.max(s / r, 0.0));
            return new Rect2d(cx - w / 2.0, cy - h / 2.0, w, h);
        }

        private static double[] convertToCenter(Rect2d box) {
            double cx = box.x + box.width / 2.0;
            double cy = box.y + box.height / 2.0;
            double s = box.width * box.height;
            double r = box.width / Math.max(box.height, 1e-6);
            return new double[]{cx, cy, s, r, 0, 0, 0};
        }
    }

    private static class Sort {
        private final List<KalmanBoxTracker> trackers = new ArrayList<>();
        private final int maxAge;
        private final int minHits;
        private final float iouThreshold;
        private int nextId = 1;

        Sort(int maxAge, int minHits, float iouThreshold) {
            this.maxAge = maxAge;
            this.minHits = minHits;
            this.iouThreshold = iouThreshold;
        }

        List<Track> update(List<Detection> detections) {
            List<Track> results = new ArrayList<>();
            for (KalmanBoxTracker tracker : trackers) {
                tracker.predict();
            }

            List<MatchedPair> matches = associateDetectionsToTrackers(detections, trackers);
            Set<Integer> matchedDetections = new HashSet<>();
            Set<Integer> matchedTrackers = new HashSet<>();

            for (MatchedPair match : matches) {
                if (match.iou < iouThreshold) {
                    continue;
                }
                KalmanBoxTracker tracker = trackers.get(match.trackerIndex);
                Detection detection = detections.get(match.detectionIndex);
                tracker.update(detection.box, detection.classId, detection.score);
                matchedDetections.add(match.detectionIndex);
                matchedTrackers.add(match.trackerIndex);
            }

            for (int i = 0; i < detections.size(); i++) {
                if (!matchedDetections.contains(i)) {
                    Detection detection = detections.get(i);
                    trackers.add(new KalmanBoxTracker(detection.box, nextId++, detection.classId, detection.score));
                }
            }

            List<KalmanBoxTracker> survivors = new ArrayList<>();
            for (int i = 0; i < trackers.size(); i++) {
                KalmanBoxTracker tracker = trackers.get(i);
                if (tracker.timeSinceUpdate < maxAge && (tracker.hitStreak >= minHits || tracker.age <= minHits)) {
                    Rect2d state = tracker.getState();
                    results.add(new Track(state, tracker.id, tracker.classId, tracker.score));
                    survivors.add(tracker);
                }
            }
            trackers.clear();
            trackers.addAll(survivors);
            return results;
        }

        private List<MatchedPair> associateDetectionsToTrackers(List<Detection> detections, List<KalmanBoxTracker> trackers) {
            List<MatchedPair> matched = new ArrayList<>();
            if (trackers.isEmpty() || detections.isEmpty()) {
                return matched;
            }
            for (int d = 0; d < detections.size(); d++) {
                double bestIou = 0.0;
                int bestT = -1;
                for (int t = 0; t < trackers.size(); t++) {
                    double iou = computeIoU(detections.get(d).box, trackers.get(t).getState());
                    if (iou > bestIou) {
                        bestIou = iou;
                        bestT = t;
                    }
                }
                if (bestT >= 0) {
                    matched.add(new MatchedPair(d, bestT, bestIou));
                }
            }
            matched.sort(Comparator.comparingDouble(m -> -m.iou));
            List<MatchedPair> filtered = new ArrayList<>();
            Set<Integer> usedDets = new HashSet<>();
            Set<Integer> usedTrks = new HashSet<>();
            for (MatchedPair pair : matched) {
                if (!usedDets.contains(pair.detectionIndex) && !usedTrks.contains(pair.trackerIndex)) {
                    usedDets.add(pair.detectionIndex);
                    usedTrks.add(pair.trackerIndex);
                    filtered.add(pair);
                }
            }
            return filtered;
        }
    }

    private static class MatchedPair {
        int detectionIndex;
        int trackerIndex;
        double iou;

        MatchedPair(int detectionIndex, int trackerIndex, double iou) {
            this.detectionIndex = detectionIndex;
            this.trackerIndex = trackerIndex;
            this.iou = iou;
        }
    }

    public static void main(String[] args) throws Exception {
        String source = "0";
        String modelPath = "yolov8n.onnx";
        String classesPath = "coco.names";
        float confThreshold = 0.35f;
        float nmsThreshold = 0.45f;
        int inputWidth = 640;
        int inputHeight = 640;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--source" -> {
                    if (i + 1 < args.length) source = args[++i];
                }
                case "--model" -> {
                    if (i + 1 < args.length) modelPath = args[++i];
                }
                case "--classes" -> {
                    if (i + 1 < args.length) classesPath = args[++i];
                }
                case "--conf" -> {
                    if (i + 1 < args.length) confThreshold = Float.parseFloat(args[++i]);
                }
                case "--nms" -> {
                    if (i + 1 < args.length) nmsThreshold = Float.parseFloat(args[++i]);
                }
                case "--width" -> {
                    if (i + 1 < args.length) inputWidth = Integer.parseInt(args[++i]);
                }
                case "--height" -> {
                    if (i + 1 < args.length) inputHeight = Integer.parseInt(args[++i]);
                }
            }
        }

        List<String> classNames = loadClassNames(classesPath);
        VideoCapture capture = new VideoCapture();
        if (source.matches("\\d+")) {
            capture.open(Integer.parseInt(source));
        } else {
            capture.open(source);
        }
        if (!capture.isOpened()) {
            System.err.println("Cannot open source: " + source);
            return;
        }

        Net net = Dnn.readNetFromONNX(modelPath);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        Sort tracker = new Sort(30, 3, 0.3f);
        Mat frame = new Mat();
        long lastTime = System.nanoTime();

        while (capture.read(frame)) {
            if (frame.empty()) {
                break;
            }

            Mat resized = new Mat();
            Size inputSize = new Size(inputWidth, inputHeight);
            Imgproc.resize(frame, resized, inputSize);
            Mat blob = Dnn.blobFromImage(resized, 1.0 / 255.0, inputSize, new Scalar(0, 0, 0), true, false);
            net.setInput(blob);
            Mat output = net.forward();

            List<Detection> detections = postprocess(frame, output, confThreshold, nmsThreshold);
            List<Track> tracks = tracker.update(detections);
            drawDetections(frame, detections, tracks, classNames);

            double fps = 1e9 / (System.nanoTime() - lastTime);
            lastTime = System.nanoTime();
            Imgproc.putText(frame, String.format("FPS: %.1f", fps), new Point(10, 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(255, 255, 255), 2);
            HighGui.imshow("Object Detection + SORT Tracking", frame);
            if (HighGui.waitKey(1) == 27) {
                break;
            }
        }

        capture.release();
        HighGui.destroyAllWindows();
    }

    private static List<Detection> postprocess(Mat frame, Mat output, float confThreshold, float nmsThreshold) {
        List<Detection> detections = new ArrayList<>();
        if (output.empty()) {
            return detections;
        }

        int rows = output.rows();
        int cols = output.cols();
        Mat outputReshaped = output.reshape(1, rows);
        float[] data = new float[cols];
        for (int i = 0; i < rows; i++) {
            outputReshaped.get(i, 0, data);
            float objectness = data[4];
            if (objectness < confThreshold) {
                continue;
            }
            float maxClassScore = 0.0f;
            int classId = -1;
            for (int c = 5; c < cols; c++) {
                if (data[c] > maxClassScore) {
                    maxClassScore = data[c];
                    classId = c - 5;
                }
            }
            float confidence = objectness * maxClassScore;
            if (confidence < confThreshold) {
                continue;
            }
            float cx = data[0] * frame.cols();
            float cy = data[1] * frame.rows();
            float w = data[2] * frame.cols();
            float h = data[3] * frame.rows();
            float left = cx - w / 2.0f;
            float top = cy - h / 2.0f;
            Rect2d box = new Rect2d(left, top, w, h);
            detections.add(new Detection(box, confidence, classId));
        }

        if (detections.isEmpty()) {
            return detections;
        }

        List<Rect2d> boxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        for (Detection detection : detections) {
            boxes.add(detection.box);
            confidences.add(detection.score);
        }
        MatOfFloat confidencesMat = new MatOfFloat();
        confidencesMat.fromList(confidences);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidencesMat, confThreshold, nmsThreshold, indices);

        List<Detection> result = new ArrayList<>();
        int[] indexArray = indices.toArray();
        for (int idx : indexArray) {
            if (idx >= 0 && idx < detections.size()) {
                result.add(detections.get(idx));
            }
        }
        return result;
    }

    private static void drawDetections(Mat frame, List<Detection> detections, List<Track> tracks, List<String> classNames) {
        for (Track track : tracks) {
            Imgproc.rectangle(frame, track.box.tl(), track.box.br(), new Scalar(0, 255, 0), 2);
            String label = "ID " + track.id;
            if (track.classId >= 0 && track.classId < classNames.size()) {
                label = classNames.get(track.classId) + " " + label;
            }
            Imgproc.putText(frame, label, new Point(track.box.x, Math.max(track.box.y - 10, 10)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 2);
        }
        for (Detection det : detections) {
            Imgproc.rectangle(frame, det.box.tl(), det.box.br(), new Scalar(255, 0, 0), 1);
        }
    }

    private static List<String> loadClassNames(String filename) throws Exception {
        List<String> names = new ArrayList<>();
        File file = new File(filename);
        if (!file.exists()) {
            return names;
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String value = line.trim();
                if (!value.isEmpty()) {
                    names.add(value);
                }
            }
        }
        return names;
    }

    private static double computeIoU(Rect2d a, Rect2d b) {
        double x1 = Math.max(a.x, b.x);
        double y1 = Math.max(a.y, b.y);
        double x2 = Math.min(a.x + a.width, b.x + b.width);
        double y2 = Math.min(a.y + a.height, b.y + b.height);
        double inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        double union = a.area() + b.area() - inter;
        return union <= 0 ? 0.0 : inter / union;
    }

    private static double[][] transpose(double[][] m) {
        int rows = m.length;
        int cols = m[0].length;
        double[][] t = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t[j][i] = m[i][j];
            }
        }
        return t;
    }

    private static double[] matMul(double[][] m, double[] v) {
        int rows = m.length;
        int cols = m[0].length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += m[i][j] * v[j];
            }
        }
        return result;
    }

    private static double[][] matMul(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = b[0].length;
        int inner = b.length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < inner; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    private static double[][] add(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    private static double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private static double[] subtract(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    private static double[][] subtract(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    private static double[][] identityMatrix(int size) {
        double[][] I = new double[size][size];
        for (int i = 0; i < size; i++) {
            I[i][i] = 1.0;
        }
        return I;
    }

    private static double[][] inverse(double[][] m) {
        int n = m.length;
        double[][] inv = new double[n][n];
        double[][] copy = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, copy[i], 0, n);
            inv[i][i] = 1.0;
        }
        for (int i = 0; i < n; i++) {
            double pivot = copy[i][i];
            if (Math.abs(pivot) < 1e-9) {
                continue;
            }
            for (int j = 0; j < n; j++) {
                copy[i][j] /= pivot;
                inv[i][j] /= pivot;
            }
            for (int k = 0; k < n; k++) {
                if (k == i) {
                    continue;
                }
                double factor = copy[k][i];
                for (int j = 0; j < n; j++) {
                    copy[k][j] -= factor * copy[i][j];
                    inv[k][j] -= factor * inv[i][j];
                }
            }
        }
        return inv;
    }
}
