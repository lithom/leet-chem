package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;

import java.util.ArrayList;
import java.util.List;

public class TopologicalDistancePairHandler implements FeaturePairHandler<TopologicalDistancePairHandler.TopologicalDistance> {

    public static class TopologicalDistance implements MolFeatureDistance {
        public final int d;
        public TopologicalDistance(int d) {
            this.d = d;
        }
    }

    private int minConsideredDistance;
    private double minConsideredDistanceImportance;
    private double minDistanceFullImportance;

    /**
     * Describes a topological distance handler.
     * The three parameters can be used to (1) omit pairs that are close,
     * and (2) linearly ramp up importance from minConsideredDistance to a distance.
     *
     * If you want to uniformly weight all distances, use (minConsideredDistance,1.0,minConsideredDistance)
     *
     * @param minConsideredDistance
     * @param minConsideredDistanceImportance
     * @param minDistanceFullImportance
     */
    public TopologicalDistancePairHandler(int minConsideredDistance, double minConsideredDistanceImportance, double minDistanceFullImportance) {
        this.minConsideredDistance = minConsideredDistance;
        this.minConsideredDistanceImportance = minConsideredDistanceImportance;
        this.minDistanceFullImportance = minDistanceFullImportance;
    }

    @Override
    public <F extends MolFeature> List<FeaturePair<TopologicalDistance, F>> detectPairs(StereoMolecule m, List<F> features) {
        List<FeaturePair<TopologicalDistance, F>> pairs = new ArrayList<>();
        for(int zi=0;zi<features.size()-1;zi++) {
            for(int zj=zi+1;zj<features.size();zj++) {
                // for simplicity we just compute distance in between center atoms. here we could go also for the
                // shortest distance in between all covered atoms..
                int di = m.getPathLength(features.get(zi).getCentralAtom(),features.get(zj).getCentralAtom());
                if(di<minConsideredDistance) {continue;}

                FeaturePair pi = new FeaturePair(new TopologicalDistance(di),features.get(zi),features.get(zj));
                pairs.add(pi);
            }
        }
        return pairs;
    }

    @Override
    public double computeDistanceImportance(TopologicalDistance da) {
        if(da.d >=minDistanceFullImportance || minConsideredDistanceImportance >= 1 ) {return 1;}
        return minConsideredDistanceImportance + (1.0-minConsideredDistanceImportance) * (da.d-minConsideredDistance) / (minDistanceFullImportance-minConsideredDistance);
    }

    @Override
    public double computeDistanceSimilarity(TopologicalDistance da, TopologicalDistance db) {
        double d =Math.abs(da.d-db.d);
        if(d<=0.1) {return 1.0;}
        double min = Math.min(da.d,db.d);
        if(min <=5 ) { if(d==1) {return 0.6;} }
        if(min <=9 ) { if(d==1) {return 0.75;} else if(d==2) {return 0.4;} }
        if(min <=12 ) { if(d==1) {return 0.9;} else if(d==2) {return 0.5;} else if(d==3) {return 0.15;} }
        if(min <=16 ) { if(d==1) {return 0.95;} else if(d==2) {return 0.8;} else if(d==3) {return 0.7;} else if(d==4) {return 0.5;} else if(d==5) return 0.25; }
        else { if(d==1) {return 0.975;} else if(d==2) {return 0.9;} else if(d==3) {return 0.8;} else if(d==4) {return 0.55;} else if(d==5) return 0.3; }

        return 0;
    }
}
