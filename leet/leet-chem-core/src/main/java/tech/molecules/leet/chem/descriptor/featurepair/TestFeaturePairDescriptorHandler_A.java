package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandler;
import com.actelion.research.chem.descriptor.DescriptorHandlerFlexophore;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TestFeaturePairDescriptorHandler_A {

    public static void main(String args[]) {

        List<StereoMolecule> mols = ChemUtils.loadTestMolecules_36FromDrugCentral();

        boolean computeFlexo = true;

        SimplePharmacophoreFeatureHandler fh = new SimplePharmacophoreFeatureHandler();
        TopologicalDistancePairHandler    ph = new TopologicalDistancePairHandler(4,0.4,10);

        FeaturePairDescriptor.FeatureAndPairSimilarityCombinator combinator = new FeaturePairDescriptor.FeatureAndPairSimilarityCombinator() {
            @Override
            public double combinedSimilarity(double simDistance, double simFeatureA, double simFeatureB) {
                return Math.pow( simDistance * simFeatureA * simFeatureB , 0.33 );
            }
        };

        DescriptorHandler dh1 = new FeaturePairDescriptor<TopologicalDistancePairHandler.TopologicalDistance,
        SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint>(fh,ph,combinator);

        List<String> mol_idcodes = mols.stream().map(mi -> mi.getIDCode()).collect(Collectors.toList());

        Map<String,Object> descriptors_flexo = new HashMap<>();
        Map<String,List<SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint>> descriptors_holo = new HashMap<>();

        DescriptorHandlerFlexophore dh2 = new DescriptorHandlerFlexophore();
        long ts_a = System.currentTimeMillis();
        if(computeFlexo) {
            for (int zi = 0; zi < mols.size(); zi++) {
                descriptors_flexo.put(mols.get(zi).getIDCode(), dh2.createDescriptor(mols.get(zi)));
            }
        }
        long ts_b = System.currentTimeMillis();
        for(int zi=0;zi<mols.size();zi++) {
            descriptors_holo.put(mols.get(zi).getIDCode(), (List<SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint>) dh1.createDescriptor(mols.get(zi)) );
        }
        long ts_c = System.currentTimeMillis();
        System.out.println("T1="+(ts_b-ts_a));
        System.out.println("T2="+(ts_c-ts_b));


        Map<Pair<String,String>,Double> sim_flexo = new HashMap<>();
        Map<Pair<String,String>,Double> sim_holo  = new HashMap<>();

        System.out.println("fp"+"\t"+"flexo"+"\t"+"a[idcode]"+"\t"+"b[idcode]");
        for(int zi=0;zi<mols.size()-1;zi++) {
            for(int zj=zi+1;zj<mols.size()-1;zj++) {
                //StereoMolecule mi = mols.get(zi);
                //StereoMolecule mj = mols.get(zj);

                List<SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint> pppa = descriptors_holo.get(mol_idcodes.get(zi));
                List<SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint> pppb = descriptors_holo.get(mol_idcodes.get(zj));
                double sa = dh1.getSimilarity(pppa,pppb);//evaluateMatch(pppa,pppb,10);
                sim_holo.put(Pair.of(mol_idcodes.get(zi),mol_idcodes.get(zj)),sa);

                double sb = Double.NaN;
                if(computeFlexo) {
                    Object flexa = descriptors_flexo.get(mol_idcodes.get(zi));
                    Object flexb = descriptors_flexo.get(mol_idcodes.get(zj));
                    sb = dh2.getSimilarity(flexa, flexb);
                }
                //System.out.println("sa="+sa+" , sb="+sb);
                System.out.println(sa+"\t"+sb+"\t"+mol_idcodes.get(zi)+"\t"+mol_idcodes.get(zj));
//                if(sa>0.75) {
//                    System.out.println("a= "+mol_idcodes.get(zi)+  "  b= "+mol_idcodes.get(zj));
//                }
            }
        }

    }

}
