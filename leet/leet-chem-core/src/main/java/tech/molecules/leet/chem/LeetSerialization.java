package tech.molecules.leet.chem;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.core.JacksonException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.util.VersionUtil;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.module.SimpleModule;

import java.io.IOException;

public class LeetSerialization {

    /**
     * Just use this one, is thread safe.s
     */
    public static final LeetObjectMapper OBJECT_MAPPER =
            new LeetObjectMapper();



    public static class LeetObjectMapper extends ObjectMapper {
        public LeetObjectMapper() {
            registerModule(new OCLModule());
        }
    }

    public static class OCLModule extends SimpleModule {
        private static final String NAME = "OpenChemLibModule";
        private static final VersionUtil VERSION_UTIL = new VersionUtil() {};

        public static final String STEREOMOLECULE_FIELD_IDC        = "M";
        public static final String STEREOMOLECULE_FIELD_IDC_COORDS = "C";
        public OCLModule() {
            super(NAME, VERSION_UTIL.version());
            addSerializer(StereoMolecule.class, new StereoMoleculeSerializer());
            addDeserializer(StereoMolecule.class, new StereoMoleculeDeserializer());
        }
    }

    public static class StereoMoleculeSerializer extends JsonSerializer<StereoMolecule> {
        @Override
        public void serialize(StereoMolecule o, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
            jsonGenerator.writeStartObject();
            Canonizer canonizer = new Canonizer(o, Canonizer.ENCODE_ATOM_CUSTOM_LABELS | Canonizer.ENCODE_ATOM_SELECTION);
            //jsonGenerator.writeStringField(OCLModule.STEREOMOLECULE_FIELD_IDC,o.getIDCode());
            //jsonGenerator.writeStringField(OCLModule.STEREOMOLECULE_FIELD_IDC_COORDS,o.getIDCoordinates());
            jsonGenerator.writeStringField(OCLModule.STEREOMOLECULE_FIELD_IDC,canonizer.getIDCode());
            jsonGenerator.writeStringField(OCLModule.STEREOMOLECULE_FIELD_IDC_COORDS,canonizer.getEncodedCoordinates(true));
            jsonGenerator.writeEndObject();
        }
    }
    public static class StereoMoleculeDeserializer extends JsonDeserializer<StereoMolecule> {

        @Override
        public StereoMolecule deserialize(JsonParser p, DeserializationContext ctxt) throws IOException, JacksonException {
            StereoMolecule mi = new StereoMolecule();
            IDCodeParser icp = new IDCodeParser();
            ObjectCodec codec = p.getCodec();
            JsonNode node = codec.readTree(p);
            String idc        = node.get(OCLModule.STEREOMOLECULE_FIELD_IDC).asText();
            String idc_coords = node.get(OCLModule.STEREOMOLECULE_FIELD_IDC_COORDS).asText("");
            icp.parse(mi,idc,idc_coords);
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            return mi;
        }

    }


}



