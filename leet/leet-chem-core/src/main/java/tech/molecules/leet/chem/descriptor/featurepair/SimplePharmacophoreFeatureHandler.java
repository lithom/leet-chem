package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.phesa.pharmacophore.IonizableGroupDetector;
import com.actelion.research.chem.phesa.pharmacophore.PharmacophoreCalculator;
import com.actelion.research.chem.phesa.pharmacophore.pp.IPharmacophorePoint;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class SimplePharmacophoreFeatureHandler implements FeatureHandler<SimplePharmacophoreFeatureHandler.PhesaPharmacophorePoint> {

    @Override
    public List<PhesaPharmacophorePoint> detectFeatures(StereoMolecule mtest) {
        List<IPharmacophorePoint> points = new ArrayList<>();
        IonizableGroupDetector detector = new IonizableGroupDetector(mtest);
        points.addAll(detector.detect());
        points.addAll(PharmacophoreCalculator.getPharmacophorePoints(mtest));

        return points.stream().map( xi -> new PhesaPharmacophorePoint(xi) ).collect(Collectors.toList());
    }

    @Override
    public double computeFeatureSimilarity(PhesaPharmacophorePoint a, PhesaPharmacophorePoint b) {
        if(a.pp.getFunctionalityIndex()==b.pp.getFunctionalityIndex()) {
            return 1;
        }
        return 0;
    }

    @Override
    public double computeFeatureImportance(PhesaPharmacophorePoint a) {
        return 1.0;
    }


    public static class PhesaPharmacophorePoint implements MolFeature {
        public final IPharmacophorePoint pp;

        public PhesaPharmacophorePoint(IPharmacophorePoint pp) {
            this.pp = pp;
        }

        @Override
        public int getCentralAtom() {
            return pp.getCenterID();
        }

        @Override
        public int[] getCoveredAtoms() {
            return new int[pp.getCenterID()];
        }
    }

}
