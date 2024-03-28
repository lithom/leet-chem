package tech.molecules.leet.chem.virtualspaces.gui.task;

import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.gui.VerticalFlowLayout;
import com.idorsia.research.chem.hyperspace.importer.MultiStepImporter;
import com.idorsia.research.chem.hyperspace.io.SynthonSpaceParser2;
import tech.molecules.leet.chem.virtualspaces.ChemicalSpaceCreator2;
import tech.molecules.leet.chem.virtualspaces.SpaceCreation_A;
import tech.molecules.leet.chem.virtualspaces.gui.BuildingBlockFile;
import tech.molecules.leet.chem.virtualspaces.gui.LoadedBB;
import tech.molecules.leet.chem.virtualspaces.gui.ReactionMechanism;
import tech.molecules.leet.chem.virtualspaces.gui.SpaceCreatorController;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class CreateSpaceTask extends AbstractAction {

    private SpaceCreatorController controller;


    private File fileOutputDirectory = null;
    private int maxAtoms = 0;

    private String importerMode = "2split";

    public CreateSpaceTask(SpaceCreatorController controller) {
        super("Create Space");
        this.controller = controller;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Map<BuildingBlockFile, List<LoadedBB>> loadedBBs = this.controller.getModel().getBuildingBlockFileTableModel().getLoadedBBs();
        List<LoadedBB> bbs = loadedBBs.values().stream().flatMap(xi -> xi.stream()).collect(Collectors.toList());
        List<ReactionMechanism> reactionMechanisms = this.controller.getModel().getReactionMechanismTableModel().getSelectedReactionMechanisms();
        List<Reaction> reactions = reactionMechanisms.stream().map(xi -> xi.getRxn()).collect(Collectors.toList());

        if(bbs.isEmpty()) {
            JOptionPane.showMessageDialog(controller.getView().getFrame(), "No Building Blocks selected.", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }
        if(reactions.isEmpty()) {
            JOptionPane.showMessageDialog(controller.getView().getFrame(), "No Reactions selected.", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // We have to run this outside of the Swing event handling thread..
        Runnable ri = new Runnable() {
            @Override
            public void run() {
                runSpaceCreationProcedure(loadedBBs,bbs,reactionMechanisms,reactions);
            }
        };
        Thread ti = new Thread(ri);
        ti.start();

    }


    private void runSpaceCreationProcedure(Map<BuildingBlockFile, List<LoadedBB>> loadedBBs, List<LoadedBB> bbs, List<ReactionMechanism> reactionMechanisms, List<Reaction> reactions) {

        JDialog dialog = createConfigureSpaceCreationDialog(controller.getView().getFrame());
        dialog.setVisible(true);

        //
        // Here we start with the actual space creation..
        //
        StatusDialog statusDialog = new StatusDialog(controller.getView().getFrame());
        statusDialog.setVisible(true);
        dialog.revalidate();
        dialog.repaint();


        List<Function<LoadedBB,Boolean>> filters = new ArrayList<>();
        if(maxAtoms > 0) {
            filters.add( (x) -> x.getNumAtoms() <= maxAtoms );
        }

        System.out.println("Space creation: Start!");
        statusDialog.addStatusMessage("Space creation: Start!");
        ChemicalSpaceCreator2 spaceCreator = SpaceCreation_A.createSpaceCreator(bbs,reactions,fileOutputDirectory,filters,(x) -> statusDialog.addStatusMessage(x));
        statusDialog.addStatusMessage("BBs: "+bbs.size() + "Rxns: "+reactions.size());
        spaceCreator.create();
        System.out.println("Space creation: DONE!");
        statusDialog.addStatusMessage("Combinatorial Library creation: DONE!");

        statusDialog.addStatusMessage("Hyperspace Import: Start!");
        System.out.println("Hyperspace Import: Start!");
        String synthonSpaceFile = fileOutputDirectory.getAbsolutePath()+File.separator+"synthonSpace.tsv";
        String[] importerArguments = new String[]{importerMode,
                fileOutputDirectory.getAbsolutePath()+File.separator+"CombinatorialLibraries",
                synthonSpaceFile,
                "BB-ID"};
        MultiStepImporter.main(importerArguments);

        System.out.println("Hyperspace Import: DONE!");
        statusDialog.addStatusMessage("Hyperspace Import: DONE!");

        System.out.println("Hyperspace Space Creation: Start!");

        int numProcessors = Math.max( 1 , Runtime.getRuntime().availableProcessors()-1 );
        statusDialog.addStatusMessage("Hyperspace Space Creation: Start! using "+numProcessors+" cores");
        SynthonSpaceParser2.initREALSpace(synthonSpaceFile,"Hyperspace","FragFp", numProcessors, fileOutputDirectory.getAbsolutePath());
        System.out.println("Hyperspace Space Creation: DONE!");
        statusDialog.addStatusMessage("Hyperspace Space Creation: DONE!");
    }

    private JDialog createConfigureSpaceCreationDialog(JFrame parentFrame) {
        // Create the directory selection dialog
        JDialog configureSpaceCreationDialog = new JDialog(parentFrame, "Create Space", true);
        configureSpaceCreationDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);

        // Create a panel to hold the components
        JPanel panel = new JPanel(new VerticalFlowLayout());

        // Create the "Select Directory" button
        JPanel panelA = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton selectDirectoryButton = new JButton("Select Directory");
        JTextField textfieldDirectory = new JTextField(48);
        textfieldDirectory.setEditable(false);
        panelA.add(selectDirectoryButton);
        panelA.add(textfieldDirectory);
        panel.add(panelA);
        selectDirectoryButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Create the file chooser
                JFileChooser fileChooser = new JFileChooser();
                fileChooser.setDialogTitle("Select Directory");
                fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

                // Show the file chooser dialog
                int userSelection = fileChooser.showDialog(configureSpaceCreationDialog, "Select");

                // Handle the selected directory
                if (userSelection == JFileChooser.APPROVE_OPTION) {
                    File selectedDirectory = fileChooser.getSelectedFile();
                    fileOutputDirectory = selectedDirectory;
                    textfieldDirectory.setText(selectedDirectory.getAbsolutePath());
                    //JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Selected directory: " + selectedDirectory.getAbsolutePath());
                }
            }
        });

        JPanel panelB = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JComboBox<String> jcbImporterMode = new JComboBox<>();
        jcbImporterMode.addItem("1split"); jcbImporterMode.addItem("2split");
        jcbImporterMode.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                importerMode = (String) jcbImporterMode.getSelectedItem();
            }
        });
        jcbImporterMode.setSelectedItem("2split");
        panelB.add(new JLabel("Importer Mode "));
        panelB.add(jcbImporterMode);
        panel.add(panelB);

        // Create the "Max Atoms" label and text field
        JPanel panelC = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel maxAtomsLabel = new JLabel("Max Atoms:");
        JTextField maxAtomsTextField = new JTextField();
        panelC.add(maxAtomsLabel);
        panelC.add(maxAtomsTextField);

        // Add the label and text field to the panel
        panel.add(panelC);

        // Create the "Start Compute" button
        JButton startComputeButton = new JButton("Start Space Creation!");
        startComputeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Get the value from the text field
                String maxAtomsText = maxAtomsTextField.getText();
                if(!maxAtomsText.isEmpty()) {
                    // Check if it's a valid integer
                    try {
                        maxAtoms = Integer.parseInt(maxAtomsText);
                        //JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Max Atoms: " + maxAtoms);
                        configureSpaceCreationDialog.dispose();
                    } catch (NumberFormatException ex) {
                        JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Please enter a valid integer for Max Atoms.", "Error", JOptionPane.ERROR_MESSAGE);
                    }
                }
            }
        });

        // Add the "Select Directory" and "Start Compute" buttons to the panel
        //panel.add(selectDirectoryButton);
        JPanel panelD = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        panelD.add(startComputeButton);
        panel.add(panelD);

        // Add the panel to the directory selection dialog
        configureSpaceCreationDialog.getContentPane().add(panel);

        // Set the size and location of the directory selection dialog
        configureSpaceCreationDialog.pack();
        configureSpaceCreationDialog.setLocationRelativeTo(parentFrame);

        return configureSpaceCreationDialog;
    }


    public static void main(String args[]) {
        StatusDialog statusDialog = new StatusDialog(null);
        statusDialog.setVisible(true);

        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                // Simulate a long-running task
                for (int i = 1; i <= 10; i++) {
                    Thread.sleep(1000); // Simulate task progress
                    publish("Status Message " + i); // Send status messages
                }
                return null;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                // Update the dialog with status messages
                for (String message : chunks) {
                    statusDialog.addStatusMessage(message);
                }
            }

            @Override
            protected void done() {
                statusDialog.addStatusMessage("Task Completed!");
            }
        };

        worker.execute();
    }

    static class StatusDialog extends JDialog {
        private JTextArea textArea;

        public StatusDialog(JFrame owner) {
            super(owner, "Space Creation", false);
            setSize(1024, 800);

            textArea = new JTextArea();
            textArea.setEditable(false);
            JScrollPane scrollPane = new JScrollPane(textArea);

            getContentPane().setLayout(new BorderLayout());
            getContentPane().add(scrollPane,BorderLayout.CENTER);
        }

        public void addStatusMessage(String message) {
            SwingUtilities.invokeLater(() -> textArea.append(message + "\n"));
            this.revalidate();
            this.repaint();
        }
    }

}
