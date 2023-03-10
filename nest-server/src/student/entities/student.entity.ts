import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { HydratedDocument } from 'mongoose';

export type StudentDocument = HydratedDocument<Student>;

@Schema()
export class Student {
  @Prop()
  stuname: string;
  @Prop()
  stuclass: string;
  @Prop()
  gender: string;
  @Prop()
  phone: string;
  // @Prop()
  // idcard: number
  // @Prop()
  // stuprofession: string
  @Prop()
  email: string;
  @Prop()
  address: Array<number>;
  @Prop()
  workplace: Array<number>;
  @Prop()
  salary: string;
  @Prop()
  job: string;
  @Prop()
  industry: Array<string>;
  @Prop()
  company: Array<string>;
}

export const StudentSchema = SchemaFactory.createForClass(Student);
