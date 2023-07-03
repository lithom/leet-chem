package tech.molecules.leet.gui.chem.project.action;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.chem.sar.SimpleSynthonSet;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class AddSeriesAction extends AbstractAction implements ObjectSpecific {
    private final Consumer<SimpleSARSeries> seriesConsumer;

    public AddSeriesAction(Consumer<SimpleSARSeries> seriesConsumer) {
        super("Add Series");
        this.seriesConsumer = seriesConsumer;
    }

    private SimpleSARSeries obj;

    @Override
    public void setObject(Object obj) {
        if(obj instanceof SimpleSARSeries) {
            this.obj = (SimpleSARSeries)obj;
        }
    }

    @Override
    public SimpleSARSeries getObject() {
        return this.obj;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        // Create dialog components
        JTextField seriesNameField = new JTextField(20);
        JButton createButton = new JButton("Create");
        JPanel dialogPanel = new JPanel(new FlowLayout());

        // Add components to the panel
        dialogPanel.add(new JLabel("Series Name:"));
        dialogPanel.add(seriesNameField);
        dialogPanel.add(createButton);

        // Create the dialog
        JDialog dialog = new JDialog((Frame) null, "Add Series", true);
        dialog.getContentPane().add(dialogPanel);
        dialog.pack();

        // Add button action listener
        createButton.addActionListener(event -> {
            String seriesName = seriesNameField.getText();
            if (!seriesName.isEmpty()) {
                SimpleMultiSynthonStructure smss = new SimpleMultiSynthonStructure();
                List<StereoMolecule> mols_a = new ArrayList<>();
                mols_a.add(new StereoMolecule());
                SimpleSynthonSet ssa = new SimpleSynthonSet(mols_a);
                List<SimpleSynthonSet> synthonSets = new ArrayList<>();
                synthonSets.add(ssa);
                smss.setSynthonSets(synthonSets);
                SimpleSARSeries newSeries = new SimpleSARSeries(seriesName, smss);
                seriesConsumer.accept(newSeries);
                dialog.dispose();
            } else {
                JOptionPane.showMessageDialog(dialog, "Series name cannot be empty", "Error", JOptionPane.ERROR_MESSAGE);
            }
        });

        // Show dialog
        dialog.setLocationRelativeTo(null);
        dialog.setVisible(true);
    }
}
