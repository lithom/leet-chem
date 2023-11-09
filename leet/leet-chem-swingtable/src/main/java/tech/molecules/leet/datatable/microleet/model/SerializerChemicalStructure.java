package tech.molecules.leet.datatable.microleet.model;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.chem.StructureWithID;

import java.util.ArrayList;
import java.util.Base64;
import java.util.BitSet;
import java.util.List;

public class SerializerChemicalStructure implements MicroLeetDataValueSerializer<StructureRecord> {

    @Override
    public StructureRecord initFromString(String data) {
        if(data.startsWith("__SID")) {
            data = data.substring(5);
            String[] splits = data.split("\\t");
            return new StructureRecord(splits[0],splits[1],new String[]{splits[2],splits[3]},deserialize(splits[4]));
        }
        StereoMolecule parsed = null;
        data = data.trim();
        if(data.contains(" ")) {
            try{
                String split[] = data.split(" ");
                parsed = ChemUtils.parseIDCode( split[0] , split[1] );
            }
            catch(Exception ex) {
            }
        }
        else {
            try{
                parsed = ChemUtils.parseIDCode( data );
            }
            catch(Exception ex) {
            }
        }
        if(parsed==null) {
            try{
                parsed = ChemUtils.parseSmiles( data );
            }
            catch(Exception ex) {
            }
        }

        return new StructureRecord( parsed );
    }

    @Override
    public String serializeToString(StructureRecord val) {
        List<String> parts = new ArrayList<>();
        parts.add(val.molid);
        parts.add(val.batchid);
        parts.add(val.structure[0]);
        parts.add(val.structure[1]);
        parts.add(serialize(val.ffp));
        return "__SID"+ String.join("\t",parts);
        //Canonizer ca = new Canonizer(val);
        //return ca.getIDCode()+" "+ca.getEncodedCoordinates();
    }

    public static String serialize(BitSet bitSet) {
        // Retrieve the internal bit representation of the BitSet
        byte[] bytes = bitSet.toByteArray();
        // Convert to a Base64 string for easy storage and transmission
        return Base64.getEncoder().encodeToString(bytes);
    }
    public static BitSet deserialize(String base64String) {
        // Decode the Base64 String to a byte array
        byte[] bytes = Base64.getDecoder().decode(base64String);
        // Convert the byte array back to a BitSet
        return BitSet.valueOf(bytes);
    }

    @Override
    public Class getRepresentationClass() {
        return StereoMolecule.class;
    }
}
