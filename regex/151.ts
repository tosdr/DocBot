import { Regex } from '../models';

module.exports = {
    	expression: new RegExp("^(?=.*must be commenced)|(?=.*cause of action)|(?=.*forever barred)|(?=.*permanently barred)", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 151,
	name: "Users have a reduced time period to take legal action against the service"
} as Regex;