import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*at least)((?=.*years old)|(?=.*years of age))|(?=.*you must be)(?=.*old)|((?=.*years of age)|(?=.*under the age)|(?=.*age is under)|(?=.*not intended for)|(?=.*only for))((?=.*13)|(?=.*thirteen)|(?=.*16)|(?=.*sixteen)|(?=.*18)|(?=.*eighteen)))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 152,
	name: "This service is only available to users over a certain age"
} as Regex;