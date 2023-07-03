package tech.molecules.leet.gui.chem.editor.sar;

// ... other imports ...
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.sar.MultiFragment;
import tech.molecules.leet.chem.sar.SARElement;
import tech.molecules.leet.chem.sar.SimpleMultiFragment;
import tech.molecules.leet.chem.sar.SimpleSARElement;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class MultiFragmentPanel extends JPanel implements SimpleMultiFragment.MultiFragmentListener {

    private JTabbedPane tabbedPane;
    private SimpleMultiFragment model;
    private List<SimpleMultiFragment.MultiFragmentListener> listeners = new ArrayList<>();

    public MultiFragmentPanel(SimpleMultiFragment model) {
        this.model = model;
        this.model.addMultiFragmentElementListener(this);


        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        setLayout(new BorderLayout());
        tabbedPane = new JTabbedPane();
        add(tabbedPane, BorderLayout.CENTER);

        for(SimpleSARElement ei : this.model.getElements()) {
            SARElementPanel spi = new SARElementPanel(ei);
            this.tabbedPane.add("Element",spi);
            spi.addSARElementListener(new SARElementPanel.SARElementListener() {
                @Override
                public void onEdit() {
                    //fireSARElementEdited(ei);
                    model.setFragmentEdited(ei);
                }

                @Override
                public void onRemove() {

                }
            });
        }

        JButton addFragmentButton = new JButton("+");
        addFragmentButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.addMultiFragmentElement(new SimpleSARElement(new StereoMolecule())); // your model should fire a property change event here
            }
        });

//        JButton removeFragmentButton = new JButton("Remove Fragment");
//        removeFragmentButton.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                int selectedTabIndex = tabbedPane.getSelectedIndex();
//                model.removeMultiFragmentElement(selectedTabIndex); // your model should fire a property change event here
//            }
//        });

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BorderLayout());
        JTextField jtf  = new JTextField();
        topPanel.add(jtf,BorderLayout.CENTER);
        topPanel.add(addFragmentButton,BorderLayout.EAST);
        //buttonPanel.add(removeFragmentButton);
        add(topPanel, BorderLayout.NORTH);
        SwingUtilities.updateComponentTreeUI(this);
    }

    // Invoked when a MultiFragmentElement is added
    @Override
    public void onMultiFragmentElementAdded(SimpleSARElement element) {
        // Add visual representation of MultiFragmentElement to the appropriate tab
        this.reinit();
    }

    // Invoked when a MultiFragmentElement is removed
    @Override
    public void onMultiFragmentElementRemoved(int index) {
        this.reinit();
    }

    @Override
    public void onSARElementSelected(SimpleSARElement si) {

    }
}

