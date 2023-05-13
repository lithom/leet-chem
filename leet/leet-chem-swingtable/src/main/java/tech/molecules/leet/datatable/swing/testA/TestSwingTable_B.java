package tech.molecules.leet.datatable.swing.testA;

import com.actelion.research.gui.table.ChemistryCellRenderer;
import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.dataprovider.CSVImporter;
import tech.molecules.leet.datatable.swing.CellRendererHelper;
import tech.molecules.leet.datatable.swing.DefaultSwingTableController;
import tech.molecules.leet.datatable.swing.DefaultSwingTableModel;

import javax.swing.*;
import java.io.File;

public class TestSwingTable_B {

    public static void main(String args[]) {

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel(new FlatLightLaf());
        } catch (Exception ex) {
            System.err.println("Failed to initialize LaF");
        }

        // Import data:
        CSVImporter importer = new CSVImporter(new File("C:\\Temp\\leet_input\\chembl_assay_b.csv"));
        CSVImporter.ImportedCSV data = importer.createTableData(5);
        DataTable dtable = CSVImporter.createDataTable(data);
        dtable.setAllKeys(data.getKeys());

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        DefaultSwingTableModel model = new DefaultSwingTableModel(dtable);
        DefaultSwingTableController tableController = new DefaultSwingTableController(model);
        tableController.setRowHeight(140);

        tableController.setTableCellRenderer(0,new ChemistryCellRenderer());
        CellRendererHelper.configureDefaultRenderers(tableController);


        fi.getContentPane().add(tableController);
        fi.setSize(600,600);
        fi.setVisible(true);
    }


}
