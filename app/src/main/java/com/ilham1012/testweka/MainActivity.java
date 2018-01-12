package com.ilham1012.testweka;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;


public class MainActivity extends AppCompatActivity {
    double sepalLength, sepalWidth, petalLength, petalWidth;
    EditText sLengthEdit, sWidthEdit, pLengthEdit, pWidthEdit;
    TextView resultText;

    private Classifier wekaClassifier = null;

    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sLengthEdit = findViewById(R.id.sLengthInput);
        sWidthEdit = findViewById(R.id.sWidthInput);
        pLengthEdit = findViewById(R.id.pLengthInput);
        pWidthEdit = findViewById(R.id.pWidthInput);
        resultText  = findViewById(R.id.resultView);
    }

    public void classify(View view) {
        // Get value from EditText
        sepalLength = Double.parseDouble(sLengthEdit.getText().toString());
        sepalWidth  = Double.parseDouble(sWidthEdit.getText().toString());
        petalLength = Double.parseDouble(pLengthEdit.getText().toString());
        petalWidth  = Double.parseDouble(pWidthEdit.getText().toString());


        // Do Calculation
        String result = classifyWeka(sepalLength, sepalWidth, petalLength, petalWidth);

        // Display Result
        resultText.setText(result);
    }

    public String classifyWeka(double sLength, double sWidth, double pLength, double pWidth){
        String classificationResult = "";
        AssetManager assetManager = getAssets();

        try{
            wekaClassifier = (Classifier) weka.core.SerializationHelper.read(assetManager.open("iris__SMO-no_norm_no_stand.model"));

            if (wekaClassifier==null){
                Toast.makeText(this, "Model not loaded!", Toast.LENGTH_SHORT).show();
            }

            final Attribute atrSepalLength = new Attribute("sepallength");
            final Attribute atrSepalWidth = new Attribute("sepalwidth");
            final Attribute atrPetalLength = new Attribute("petallength");
            final Attribute atrPetalWidth = new Attribute("petalwidth");
            final List<String> classes = new ArrayList<String>() {
                {
                    add("Iris-setosa"); // cls nr 1
                    add("Iris-versicolor"); // cls nr 2
                    add("Iris-virginica"); // cls nr 3
                }
            };

            ArrayList<Attribute> attributeList = new ArrayList<Attribute>(2){
                {
                    add(atrSepalLength);
                    add(atrSepalWidth);
                    add(atrPetalLength);
                    add(atrPetalWidth);
                    Attribute atrClass = new Attribute("@@class@@", classes);
                    add(atrClass);
                }
            };

            Instances trainingSet = new Instances("Rel", attributeList, 1);
            trainingSet.setClassIndex(4);

            DenseInstance instance = new DenseInstance(5);
            instance.setValue(atrSepalLength, sLength);
            instance.setValue(atrSepalWidth, sWidth);
            instance.setValue(atrPetalLength, pLength);
            instance.setValue(atrPetalWidth, pWidth);

            instance.setDataset(trainingSet);

            double prediction = wekaClassifier.classifyInstance(instance);

            classificationResult = "Result : " + classes.get(new Double(prediction).intValue()) + " - " + prediction;

        } catch (IOException e){
            e.printStackTrace();
        } catch (Exception e){
            e.printStackTrace();
        }

        return classificationResult;
    }
}














