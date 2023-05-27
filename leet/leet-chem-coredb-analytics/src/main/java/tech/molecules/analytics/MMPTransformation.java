package tech.molecules.analytics;

import tech.molecules.leet.chem.shredder.FragmentDecomposition;

public interface MMPTransformation extends Comparable<MMPTransformation> {

    public String getFragmentAId();
    public String getFragmentBId();


    public default String getTransformationId() {
        return getFragmentAId()+":::::"+getFragmentBId();
    }

    public default String getInverseTransformationId() {
        return getFragmentBId()+":::::"+getFragmentAId();
    }

    public MMPTransformation getInverseTransformation();

    public default String getTransformationIdWithoutDirection() {
        String fa = getFragmentAId();
        String fb = getFragmentBId();

        return (fa.compareTo(fb) <= 0) ? getTransformationId()
                                       : getInverseTransformationId();
    }

    @Override
    default int compareTo(MMPTransformation o) {
        return this.getTransformationId().compareTo(o.getTransformationId());
    }
}
