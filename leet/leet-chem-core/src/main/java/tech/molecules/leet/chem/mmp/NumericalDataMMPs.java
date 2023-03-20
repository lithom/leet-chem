package tech.molecules.leet.chem.mmp;

import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class NumericalDataMMPs {

    @JsonPropertyDescription("name of the analyzed property")
    @JsonProperty("property")
    String property;

    @JsonPropertyDescription("idcodes of analyzed structures")
    @JsonProperty("structures")
    List<String> structures;

    @JsonPropertyDescription("ids of analyzed structures")
    @JsonProperty("structureIds")
    List<String> structureIds;

}
