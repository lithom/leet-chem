package tech.molecules.leet.datatable;


import javax.swing.table.TableModel;
import java.io.*;
import java.util.Map;

/**
 *
 * Different parts to serialize:
 * 1. DataProviders
 * 2. AllKeys (List of strings)
 * 3. Columns
 * 4. Filters / Sorts on top of columns
 * 5. SelectionModel
 * 6. NumericDatasources / CategoricDatasources
 *
 *
 * Columns parts in detail:
 * background / numeric datasources
 *
 *
 * Idea of serialization is as follows:
 *
 * 1. Columns should be serializable and serialize everything except listeners
 * 2. DataProviders will be serialized via a special procedure
 * 3. DataFilter and DataSort should also be serializable and serialize everything except listeners
 * 4. SelectionModel
 *
 *
 */
public class DataTableSerializer {

    public void serialize(DataTable table) {




        ByteArrayOutputStream datatable_out = new ByteArrayOutputStream();

        //String filename = "datatable1.data";
        try (ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(datatable_out))) {
            out.writeObject(table);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // now serialize the dataproviders..
        Map<DataTableColumn,DataProvider> dataProviderStructure = table.getDataTableProviderStructure();
        // TODO: implement serialization of this..


    }

    public DataTable deserializeDataTable(String filenameDataTable) {
        DataTable tm = null;
        try (ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filenameDataTable)))) {
             tm = (DataTable) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
        return tm;
    }



}
