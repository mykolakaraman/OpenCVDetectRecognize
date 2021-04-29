package org.astral.text.detect.recognize;

import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import nu.pattern.OpenCV;
import org.apache.commons.io.IOUtils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.TextDetectionModel_EAST;
import org.opencv.dnn.TextRecognitionModel;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


/**
 * Originally C++ converted to Java:
 * https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
 */
public class OpenCVEastDetector {

    public static void main(String... args) throws Exception {
        OpenCV.loadShared();
        OpenCVEastDetector detector = new OpenCVEastDetector();

        detector.processFile("Witcher4.jpg");
    }

    public void processFile(String name) throws Exception {
        final String absolutPath = getClass().getClassLoader().getResource("photos/witcher/" + name).getPath();

        // Convert Image to Mat before processing
        Mat srcMat = Imgcodecs.imread(absolutPath);
        matToFile("1.read", srcMat);
        process(srcMat);
    }

    public void process(Mat matrix) throws Exception {
        float confThreshold = 0.5f; //Confidence threshold
        float nmsThreshold = 0.4f; // Non-maximum suppression threshold
        int width = 320; //Preprocess input image by resizing to a specific width. It should be multiple by 32
        int height = 320; //Preprocess input image by resizing to a specific height. It should be multiple by 32
        boolean isMatRGB = true; // in case of false it is gray scale, in case of true it is colorful

//        Path to a binary .pb file contains trained detector network
//        https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/blob/master/frozen_east_text_detection.pb
        String detModelPath = OpenCVEastDetector.class.getClassLoader().getResource("opencv/models/detector/frozen_east_text_detection.pb").getPath();
//          Path to a binary .onnx file contains trained CRNN text recognition model
//        https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
        String recModelPath = OpenCVEastDetector.class.getClassLoader().getResource("opencv/models/recognition/vgg_ctc_float16.onnx").getPath();
//        Path to benchmarks for evaluation
        URL vocPath = OpenCVEastDetector.class.getClassLoader().getResource("opencv/models/recognition/alphabet.txt");

        // Load networks.
        TextDetectionModel_EAST detector = new TextDetectionModel_EAST(detModelPath);
        detector.setConfidenceThreshold(confThreshold)
                .setNMSThreshold(nmsThreshold)
//        Issue happens here dump is located under
//        hs_err_pid76036.info
//
                ;

        TextRecognitionModel recognizer = new TextRecognitionModel(recModelPath);

        // Load vocabulary
        final List<String> vocabulary = IOUtils.readLines(vocPath.openStream(), StandardCharsets.UTF_8);
        recognizer.setVocabulary(vocabulary);
        recognizer.setDecodeType("CTC-greedy");

        // Parameters for Recognition
        double recScale = 1.0 / 127.5;
        Scalar recMean = new Scalar(127.5, 127.5, 127.5);
        Size recInputSize = new Size(100, 32);
        recognizer.setInputParams(recScale, recInputSize, recMean);

        // Parameters for Detection
        double detScale = 1.0;
        Size detInputSize = new Size(width, height);
        Scalar detMean = new Scalar(123.68, 116.78, 103.94);
        boolean swapRB = true;
        detector.setInputParams(detScale, detInputSize, detMean, swapRB);

        //process
        List<MatOfPoint> detResults = new ArrayList<>();
//        Happens if comment line 69:  .setNMSThreshold(nmsThreshold)
//        org.opencv.core.CvException: cv::Exception: OpenCV(4.5.1) /Users/runner/work/opencv/opencv/opencv-4.5.1/modules/dnn/src/model.cpp:96: error: (-201:Incorrect size of input array) Input size not specified in function 'processFrame'
        detector.detect(matrix, detResults);

        if (detResults.size() > 0) {
            // Text Recognition
            Mat recInput = new Mat();
            if (!isMatRGB) {
                Imgproc.cvtColor(matrix, recInput, Imgproc.COLOR_BGR2GRAY);
            } else {
                recInput = matrix;
            }
            List<MatOfPoint> contours = new ArrayList<>();
            for (int i = 0; i < detResults.size(); i++) {
                MatOfPoint quadrangle = detResults.get(i);
                contours.add(quadrangle);


                Mat cropped = new Mat();
                fourPointsTransform(recInput, quadrangle, cropped);
                matToFile("9.recognition.", cropped);

                String recognitionResult = recognizer.recognize(cropped);

//                Puts text on image where it was located (?)
                Imgproc.putText(matrix, recognitionResult, quadrangle.toArray()[3], FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0, 0, 255), 2);
                matToFile("10.Putting text.", matrix);
            }
            Imgproc.polylines(matrix, contours, true, new Scalar(0, 255, 0), 2);
            matToFile("11.Putting text.", matrix);
        } else {
            System.out.println("No result");
        }
    }

    static void fourPointsTransform(Mat frame, MatOfPoint vertices, Mat result) {
        Size outputSize = new Size(100, 32);

        MatOfPoint2f targetVertices = new MatOfPoint2f(
                new Point(0, outputSize.height - 1),
                new Point(0, 0),
                new Point(outputSize.width - 1, 0),
                new Point(outputSize.width - 1, outputSize.height - 1));
        Mat rotationMatrix = Imgproc.getPerspectiveTransform(vertices, targetVertices);

        Imgproc.warpPerspective(frame, result, rotationMatrix, outputSize);
    }

    private File matToFile(String prefix, Mat matrix) throws Exception {
        File file = File.createTempFile(prefix, ".jpg");

        final BufferedImage bufferedImage = matToBufferedImage(matrix);
        ImageIO.write(bufferedImage, "jpg", file);
        return file;
    }

    private BufferedImage matToBufferedImage(Mat matrix) throws Exception {
        MatOfByte mob = new MatOfByte();
        Imgcodecs.imencode(".jpg", matrix, mob);
        byte ba[] = mob.toArray();
        BufferedImage bi = ImageIO.read(new ByteArrayInputStream(ba));
        return bi;
    }

}
