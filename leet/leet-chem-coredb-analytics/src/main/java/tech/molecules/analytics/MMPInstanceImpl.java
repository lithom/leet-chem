package tech.molecules.analytics;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class MMPInstanceImpl implements MMPInstance {


    @JsonPropertyDescription("frag a, i.e. the fragment with the alphanumerically lower synthon IDCode")
    @JsonProperty("a")
    public final MMPFragmentDecomposition a;

    @JsonPropertyDescription("frag b, i.e. the fragment with the alphanumerically higher synthon IDCode")
    @JsonProperty("b")
    public final MMPFragmentDecomposition b;

    public MMPInstanceImpl(MMPFragmentDecomposition a, MMPFragmentDecomposition b) {

        if(a.getDecompositionSynthon().getSynthonIDCode().compareTo(b.getDecompositionSynthon().getSynthonIDCode())<=0) {
            this.a = a;
            this.b = b;
        }
        else {
            this.a = b;
            this.b = a;
        }

    }

    @Override
    public MMPTransformation getTransformation() {
        String frag_a_id = a.getDecompositionSynthon().getSynthonIDCode();
        String frag_b_id = b.getDecompositionSynthon().getSynthonIDCode();
        MMPTransformationImpl transformation = new MMPTransformationImpl(frag_a_id,frag_b_id);
        return transformation;
    }

    @Override
    public MMPFragmentDecomposition getFragmentDecompositionA() {
        return this.a;
    }

    @Override
    public MMPFragmentDecomposition getFragmentDecompositionB() {
        return this.b;
    }


    @Override
    public MMPInstance getInverseMMPInstance() {
        return new MMPInstanceImpl(this.b,this.a);
    }
}
