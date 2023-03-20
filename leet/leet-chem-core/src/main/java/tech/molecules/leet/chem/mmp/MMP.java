package tech.molecules.leet.chem.mmp;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;

public class MMP {

    @JsonPropertyDescription("frag a, i.e. the fragment with the alphanumerically lower synthon IDCode")
    @JsonProperty("a")
    public final FragmentDecompositionSynthon a;

    @JsonPropertyDescription("frag b, i.e. the fragment with the alphanumerically higher synthon IDCode")
    @JsonProperty("b")
    public final FragmentDecompositionSynthon b;

    public MMP(FragmentDecompositionSynthon a, FragmentDecompositionSynthon b) {
        this.a = a;
        this.b = b;
    }

    public static void computeMMPs() {

    }


}

