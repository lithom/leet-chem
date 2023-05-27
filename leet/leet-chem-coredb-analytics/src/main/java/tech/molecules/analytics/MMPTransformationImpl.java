package tech.molecules.analytics;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

public class MMPTransformationImpl implements MMPTransformation {

    private final String fragmentAId;
    private final String fragmentBId;

    private final int hashcode;

    public MMPTransformationImpl(String fragmentAId, String fragmentBId) {
        this.fragmentAId = fragmentAId;
        this.fragmentBId = fragmentBId;

        this.hashcode = getTransformationId().hashCode();
    }

    //@Override
    //public String getTransformationId() {
    //    return fragmentAId+":::::"+fragmentBId;
    //}

    @Override
    public String getFragmentAId() {
        return fragmentAId;
    }

    @Override
    public String getFragmentBId() {
        return fragmentBId;
    }

    @Override
    public MMPTransformation getInverseTransformation() {
        return new MMPTransformationImpl(fragmentBId,fragmentAId);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o == null || getClass() != o.getClass()) return false;

        MMPTransformationImpl that = (MMPTransformationImpl) o;

        return this.getTransformationId().equals(that.getTransformationId());
    }

    @Override
    public int hashCode() {
        return hashcode;
    }
}
