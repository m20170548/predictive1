package algorithms;

import java.util.ArrayList;
import java.util.Random;

import main.Individual;
import main.Population;
import programElements.Addition;
import programElements.Constant;
import programElements.InputVariable;
import programElements.Multiplication;
import programElements.Operator;
import programElements.ProgramElement;
import programElements.ProtectedDivision;
import programElements.Subtraction;

public class Initializer {

	private Random r;
	private ArrayList<ProgramElement> functionSet, terminalSet, fullSet;

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Constructor: create function, terminal and full set
	//
	public Initializer(int dimensionality) {
		r = new Random();

		// create function set
		functionSet = new ArrayList<ProgramElement>();
		functionSet.add(new Addition());
		functionSet.add(new Subtraction());
		functionSet.add(new Multiplication());
		functionSet.add(new ProtectedDivision());

		// create terminal set
		terminalSet = new ArrayList<ProgramElement>();
		double[] constants = { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 };
		for (int i = 0; i < constants.length; i++)
			terminalSet.add(new Constant(constants[i]));

		// add input features
		for (int i = 0; i < dimensionality; i++)
			terminalSet.add(new InputVariable(i));

		// merge
		fullSet = new ArrayList<ProgramElement>();
		for (ProgramElement programElement : functionSet)
			fullSet.add(programElement);
		for (ProgramElement programElement : terminalSet)
			fullSet.add(programElement);
	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Initialization Algorithms:
	//
	public Population rampedHalfAndHalfInitialization(int populationSize, int maxDepth) {
		int individualsPerDepth = populationSize / maxDepth;
		int remainingIndividuals = populationSize % maxDepth;
		Population population = new Population();
		int fullIndividuals, growIndividuals;

		for (int depth = 1; depth <= maxDepth; depth++) {
			if (depth == maxDepth) {
				fullIndividuals = (int) Math.floor((individualsPerDepth + remainingIndividuals) / 2.0);
				growIndividuals = (int) Math.ceil((individualsPerDepth + remainingIndividuals) / 2.0);
			} else {
				fullIndividuals = (int) Math.floor(individualsPerDepth / 2.0);
				growIndividuals = (int) Math.ceil(individualsPerDepth / 2.0);
			}

			for (int i = 0; i < fullIndividuals; i++)
				population.addIndividual(full(depth));

			for (int i = 0; i < growIndividuals; i++)
				population.addIndividual(grow(depth));
		}
		return population;
	}

	public Individual full(int maximumTreeDepth) {
		Individual individual = new Individual();
		fullInner(individual, 0, maximumTreeDepth);
		individual.setDepth(maximumTreeDepth);
		return individual;
	}

	private void fullInner(Individual individual, int currentDepth, int maximumTreeDepth) {
		if (currentDepth == maximumTreeDepth) {
			ProgramElement randomTerminal = terminalSet.get(r.nextInt(terminalSet.size()));
			individual.addProgramElement(randomTerminal);
		} else {
			Operator randomOperator = (Operator) functionSet.get(r.nextInt(functionSet.size()));
			individual.addProgramElement(randomOperator);
			for (int i = 0; i < randomOperator.getArity(); i++) {
				fullInner(individual, currentDepth + 1, maximumTreeDepth);
			}
		}
	}

	public Individual grow(int maximumTreeDepth) {
		Individual individual = new Individual();
		growInner(individual, 0, maximumTreeDepth);
		individual.calculateDepth();
		return individual;
	}

	private void growInner(Individual individual, int currentDepth, int maximumTreeDepth) {
		if (currentDepth == maximumTreeDepth) {
			ProgramElement randomTerminal = terminalSet.get(r.nextInt(terminalSet.size()));
			individual.addProgramElement(randomTerminal);
		} else {
			if (r.nextBoolean()) {
				Operator randomOperator = (Operator) functionSet.get(r.nextInt(functionSet.size()));
				individual.addProgramElement(randomOperator);
				for (int i = 0; i < randomOperator.getArity(); i++) {
					growInner(individual, currentDepth + 1, maximumTreeDepth);
				}
			} else {
				ProgramElement randomTerminal = terminalSet.get(r.nextInt(terminalSet.size()));
				individual.addProgramElement(randomTerminal);
			}
		}
	}
}
