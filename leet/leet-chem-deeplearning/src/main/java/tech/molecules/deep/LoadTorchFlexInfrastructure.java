package tech.molecules.deep;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.dim.NDIndexSlice;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.pytorch.engine.*;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.pytorch.Module;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LoadTorchFlexInfrastructure {


    //public static Module createDecoderModule() {
        //Module mi = Module.load("C:\\Users\\Thomas\\PythonProjects\\TorchFlex\\autoencoder_hist_d16_module.pt");
        //System.out.println("mkay");
        //return mi;
    //}

    public static Predictor<float[], float[]> createDecoder() {
        //Model model = Model.newInstance("C:\\Users\\Thomas\\PythonProjects\\TorchFlex\\autoencoder_hist_d16_script.pt", Device.cpu(), "PyTorch");
        Path modelDir = Paths.get("C:\\Users\\Thomas\\PythonProjects\\TorchFlex\\");
        Model mi = Model.newInstance("autoencoder_hist_d16_script.pt", Device.cpu(), "PyTorch");

        try {
            mi.load(modelDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
        System.out.println("mkay");
        Translator<float[],float[]> translator_a = new Translator<float[], float[]>() {
            @Override
            public float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
                return ndList.get(0).toFloatArray();
            }

            @Override
            public NDList processInput(TranslatorContext translatorContext, float[] doubles) throws Exception {

                NDArray a = translatorContext.getNDManager().create(doubles);
                a = a.reshape(1,a.size());
                NDList inputs = new NDList();
                inputs.add(a);
                return inputs;
            }
        };


        float[] xa = new float[100];
        Random r = new Random();
        for(int zi=0;zi<100;zi++){ xa[zi] = (float) r.nextGaussian(); }
        NDList na = new NDList();
        NDArray a = mi.getNDManager().create(xa);
        na.add(a);
        NDList nd_out = mi.getBlock().forward(null,na,false);
        System.out.println("mkay");
        Predictor<float[],float[]> pa = mi.newPredictor(translator_a);
        return pa;
    }

    public static void main(String args[]) {

        Path modelDir = Paths.get("C:\\Users\\Thomas\\PythonProjects\\TorchFlex\\");
        Model m_decoder     = Model.newInstance("autoencoder_hist_decoder_d16.pt", Device.cpu(), "PyTorch");
        Model m_transformer = Model.newInstance("model_tf_seq2latent_d16.pt", Device.cpu(), "PyTorch");

        try {
            m_decoder.load(modelDir);
            m_transformer.load(modelDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }

        System.out.println("mkay");

        //System.out.println("");
        //Predictor<float[],float[]> pa = createDecoder();
        //float[] xa = new float[100];
        Random r = new Random();
        //for(int zi=0;zi<100;zi++){ xa[zi] = (float) r.nextGaussian(); }
        //float[] result = new float[0];

        System.out.println("mkay");

        Translator<float[][],float[]> translator_a = new Translator<float[][], float[]>() {
            @Override
            public float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
                return ndList.head().toFloatArray();
            }

            @Override
            public NDList processInput(TranslatorContext translatorContext, float[][] doubles) throws Exception {
                NDArray a = translatorContext.getNDManager().create(doubles);
                //a = a.reshape(1,a.size());
                NDList inputs = new NDList();
                inputs.add(a);
                return inputs;
            }
        };

        Translator<float[],float[]> translator_b = new Translator<float[],float[]>() {
            @Override
            public float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
                return ndList.get(0).toFloatArray();
            }

            @Override
            public NDList processInput(TranslatorContext translatorContext, float[] ndlist) throws Exception {
                NDArray a = translatorContext.getNDManager().create(ndlist);
                //a = a.reshape(1,a.size());
                NDList inputs = new NDList();
                inputs.add(a);
                return inputs;
            }
        };


        Predictor<float[][],float[]> tf      = m_transformer.newPredictor(translator_a);
        Predictor<float[],float[]> decoder = m_decoder.newPredictor(translator_b);


        // create some test input:
        //Random r = new Random();
        List<float[][]> samples = new ArrayList<>();
        for(int zi=0;zi<200;zi++) {
            int li = r.nextInt(32);
            float[][] xi = new float[32][256];
            for(int zx=0;zx<li;zx++) { int pi = r.nextInt(180); xi[zx][pi] = 1f;}
            samples.add(xi);
        }

        long ta = System.currentTimeMillis();
        for(int zi=0;zi<samples.size();zi++){
            try {
                float[] tf_out     = tf.predict(samples.get(zi));
                float[] hist_out  = decoder.predict(tf_out);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
        long tb = System.currentTimeMillis();
        System.out.println("time= "+(tb-ta));


        // INSANE, THIS WORKED!!
        // Lets see if we can do this batched..

        SequentialBlock sb = new SequentialBlock();
        sb.add(m_transformer.getBlock());
        sb.add(m_decoder.getBlock());



        Translator<float[][],float[]> translator_a2 = new Translator<float[][], float[]>() {
            @Override
            public float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
                return ndList.head().toFloatArray();
            }
            @Override
            public NDList processInput(TranslatorContext translatorContext, float[][] doubles) throws Exception {
                NDArray a = translatorContext.getNDManager().create(doubles);
                //a = a.reshape(1,a.size());
                NDList inputs = new NDList();
                inputs.add(a);
                return inputs;
            }
        };

        Model model = Model.newInstance("hist_predictor");
        model.setBlock(sb);

        Predictor<float[][],float[]> sequential = model.newPredictor(translator_a2);
        System.out.println("mkay");

        List<float[]> outputs = new ArrayList<>();

        long ta2 = System.currentTimeMillis();
        for(int zi=0;zi<samples.size();zi++){
            try {
                float[] hist_out_pre     = sequential.predict(samples.get(zi));
                float[] hist_out = new float[hist_out_pre.length];
                for(int zx=0;zx<hist_out.length;zx++) {hist_out[zx] = (float)Math.exp(hist_out_pre[zx]);}
                outputs.add(hist_out);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
        long tb2 = System.currentTimeMillis();
        System.out.println("time= "+(tb2-ta2));
        System.out.println("mkay");

        //System.out.println("result:"+result);

        //F*CK YEAH!! but now lets try to calculate all of them at once

        long ta4 = System.currentTimeMillis();
        List<float[]> output_hists = null;
        try {
            output_hists = sequential.batchPredict( samples );
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        long tb4 = System.currentTimeMillis();
        System.out.println(output_hists.size());
        System.out.println("time="+(tb4-ta4));
        System.out.println("mkay");

        if(false) { // does not work for the moment..

            Translator<float[][][], float[][]> translator_a3 = new Translator<float[][][], float[][]>() {
                @Override
                public float[][] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
                    float out_a[] = ndList.get(0).toFloatArray();
                    float[][] out = new float[out_a.length / 100][100];
                    for (int zi = 0; zi < out_a.length / 100; zi++) {
                        for (int zj = 0; zj < 100; zj++) {
                            out[zi][zj] = (float) Math.exp(out_a[zi * 100 + zj]);
                        }
                    }
                    return out;
                }

                @Override
                public NDList processInput(TranslatorContext translatorContext, float[][][] doubles) throws Exception {
                    NDArray in_a = translatorContext.getNDManager().create(new Shape(new long[]{doubles.length, doubles[0].length, doubles[0][0].length}), DataType.FLOAT32);
                    for (int zi = 0; zi < doubles.length; zi++) {
                        NDIndex ndi = new NDIndex();
                        ndi.addIndices(zi);
                        ndi.addSliceDim(0, 32);
                        ndi.addSliceDim(0, 256);
                        in_a.set(ndi, translatorContext.getNDManager().create(doubles[zi]));
//                    in_a.set(new NDIndexSlice());
//                    for(int zj=0;zj<doubles[0].length;zj++) {
//                        for(int zk=0;zk<doubles[0][0].length;zk++) {
//                            in_a.setScalar(new NDIndex(zi,zj,zk),doubles[zi][zj][zk]);
//                        }
//                    }
                    }
                    //a = a.reshape(1,a.size());
                    NDList inputs = new NDList();
                    inputs.add(in_a);
                    return inputs;
                }
            };


            Predictor<float[][][], float[][]> sequential_chained = model.newPredictor(translator_a3);
            System.out.println("mkay");

            long ta3 = System.currentTimeMillis();

            float[][][] input_alltogether = new float[samples.size()][32][256];
            for (int zi = 0; zi < input_alltogether.length; zi++) {
                for (int zj = 0; zj < 32; zj++) {
                    for (int zk = 0; zk < 256; zk++) {
                        input_alltogether[zi][zj][zk] = samples.get(zi)[zj][zk];
                    }
                }
            }

            float[][] outputs_2 = new float[0][];
            try {
                outputs_2 = sequential_chained.predict(input_alltogether);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }

            //System.out.println(outputs_2);
            long tb3 = System.currentTimeMillis();
            System.out.println("time= " + (tb3 - ta3));
            System.out.println("mkay");
        }

    }

}
