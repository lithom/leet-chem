package tech.molecules.leet.datatable.swing.testA;

import com.actelion.research.gui.table.ChemistryCellRenderer;
import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.chem.sar.SimpleSARDecompositionModel;
import tech.molecules.leet.datatable.swing.CellRendererHelper;
import tech.molecules.leet.datatable.swing.DefaultSwingTableController;
import tech.molecules.leet.datatable.swing.DefaultSwingTableModel;
import tech.molecules.leet.datatable.swing.chem.SimpleSARTableModel;
import tech.molecules.leet.gui.chem.editor.SARDecompositionEditor;

import javax.swing.*;
import java.awt.*;

public class TestSwingTable_C {



    public static void main(String args[]) {

        SARDecompositionEditor editor = new SARDecompositionEditor();

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel(new FlatLightLaf());
        } catch (Exception ex) {
            System.err.println("Failed to initialize LaF");
        }

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        fi.getContentPane().add(editor, BorderLayout.CENTER);
        fi.setSize(600,600);
        fi.setVisible(true);

        editor.addDecompositionListener(new SARDecompositionEditor.DecompositionListener() {
            @Override
            public void decompositionChanged(SimpleSARDecompositionModel decompModel) {
                SimpleSARTableModel tableModel = new SimpleSARTableModel();
                tableModel.reinitTable(decompModel,decompModel.getSeries().get(0));

                DefaultSwingTableModel swingTableModel = new DefaultSwingTableModel(tableModel.getDataTable());
                DefaultSwingTableController swingTableController = new DefaultSwingTableController(swingTableModel);
                //swingTableController.setTableCellRenderer(0,new ChemistryCellRenderer());
                CellRendererHelper.configureDefaultRenderers(swingTableController);
                swingTableController.setRowHeight(120);

                JFrame fb = new JFrame(); fb.getContentPane().setLayout(new BorderLayout());
                fb.getContentPane().add(swingTableController,BorderLayout.CENTER);
                fb.setSize(400,400);
                fb.setVisible(true);
            }
        });


    }


}
