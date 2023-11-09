package tech.molecules.leet.datatable.microleet.model;

import com.actelion.research.util.datamodel.DoubleArray;
import tech.molecules.leet.data.NumericArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class SerializerMultiNumeric implements MicroLeetDataValueSerializer<NumericArray> {

    @Override
    public NumericArray initFromString(String data) {
        if(data==null) {return new NumericArray(new double[0]);}
        List<Double> values = new ArrayList<>();
        data = data.trim();
        if(data.isEmpty()) {
            return new NumericArray(new double[0]);
        }
        String split[] = data.split(";");
        //boolean all_numbers = true;
        for(String si : split) {
            try {
                si = si.trim();
                double v = Double.parseDouble(si);
                values.add(v);
            } catch (Exception ex) {
                // ok.. try once more..
                try{
                    si = si.replace('<',' ');
                    si = si.replace('>',' ');
                    double v = Double.parseDouble(si);
                    values.add(v);
                }
                catch(Exception ex2) {
                    // ok now we are done..
                    //all_numbers = false;
                    return null;
                }
            }
        }
        return new NumericArray(values.stream().mapToDouble( xi -> xi ).toArray());
    }

    @Override
    public String serializeToString(NumericArray val) {
        if(val==null) {return "";}
        List<String> vals = Arrays.stream(val.getData()).mapToObj( xi -> ""+xi ).collect(Collectors.toList());
        return String.join(";",vals);
    }

    @Override
    public Class getRepresentationClass() {
        return NumericArray.class;
    }

}